from typing import Callable
from dataclasses import dataclass

import torch
import numpy as np
from torch import optim

from .history import History
from .callback import callback as Callback
from .event import Event
from ..distributed.distributor import Distributor

from ...utils.progress import Progress

DISTRIBUTED_MODE = "distributed"
STANDALONE_MODE = "standalone"

TRAINER_INIT = "init"
TRAINER_RUNNING = "running"
TRAINER_TERMINATED = "terminated"


@dataclass
class TrainerStatus:
    UNSET: str = "unset"
    INIT: str = "init"
    RUNNING: str = "running"
    TERMINATED: str = "terminated"


@dataclass
class TrainerState:
    module: torch.nn.Module = None
    loader: torch.utils.data.DataLoader = None
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    step: Callable = None
    histories: [History] = None
    status: str = TrainerStatus.UNSET
    distributor: Distributor = None
    event: Event = None



class Trainer:
    """trainer for training models
    """
    def __init__(self, model, loader = None, optimizer = None, loss = None, keep_history = None,
                 early_stopping = None):
        """initialization

        Args:
            model (nn.Module): model will be trained
            loader (torch.DataLoader): training data loader
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            loss (Callable): could be called as 'loss(y_hat, y)'
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            keep_history (int): keep the last n-th epoch logs, `None` will keep all
        """

        step = self._get_step(model)
        
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)

        self.loss = loss
        
        event = Event()

        # set default early stopping
        if early_stopping is None:
            from .earlystop import loss_stopping
            early_stopping = loss_stopping()
        
        if early_stopping is not False:
            event.register("earlystop:check", early_stopping)

        from collections import deque
        histories = deque(maxlen = keep_history)

        self.state = TrainerState(
            module = model,
            loader = loader,
            optimizer = optimizer,
            scheduler = None,
            step = step,
            histories = histories,
            status = TrainerStatus.INIT,
            event = event,
        )


    @property
    def module(self):
        return self.state.module

    @property
    def loader(self):
        return self.state.loader

    @property
    def optimizer(self):
        return self.state.optimizer    
    
    @property
    def status(self):
        return self.state.status
    

    @property
    def histories(self):
        return self.state.histories
    
    @property
    def event(self):
        return self.state.event

    def terminate(self):
        self.state.status = TrainerStatus.TERMINATED
    

    def run(self):
        self.state.status = TrainerStatus.RUNNING
    
    def _get_step(self, module):
        if hasattr(module, 'fit_step'):
            return type(module.fit_step.__self__).fit_step
        
        return None
    

    def fit_step(self, func):
        self.state.step = func
        
        return func


    # initialize enviroment setting
    def distributed(self, address = None, workers = 4, gpu = False, **kwargs):
        '''setting distribution enviroment and initial a ray cluster connection

        Args: 
            address (string): the head of ray cluster address
            workers (int): compute task's resource
            gpu (Booleans): whether use GPU, "True" or "False"
        '''
        # self._mode = DISTRIBUTED_MODE
        # self._workers = workers
        # self._gpu = gpu
        
        # import ray
        # if not ray.is_initialized():
        #     ray.init(address = address)

        # TODO: init distributor
        from ..distributed.distributor import Distributor

        distributor = Distributor(size = workers, **kwargs)
        self.state.distributor = distributor
    

    def _train(self, config: dict):
        """distribut training details about prepare model and datasets
        Args:
            config (dict): the parameter about lr, epoch , callback , backward_rounds
        """
        # setup running state
        self.run()

        epoch = config.get("epoch", 10)
        start = config.get("start", 0)
        callback = config.get("callback", [])
        backward_rounds = config.get("backward_rounds", 1)

        if not isinstance(callback, list):
            callback = [callback]
        
        # setup callbacks
        for c in callback:
            self.register("epoch:end", c)

        loader = self.state.loader
        model = self.state.module

        if self.state.distributor is not None:
            import ray.train as train
            # TODO prepare loader and model
            loader = train.torch.prepare_data_loader(loader)
            model = train.torch.prepare_model(model)
            # TODO: remove this patch for dist
            # model.fit_step = self.model.fit_step
            # model.state_dict = self.model.state_dict
            # model.log = self.model.log
       
        train_loop(self, model, loader, epoch = epoch, start = start, backward_rounds = backward_rounds)
        

    def train(self, loader = None, epoch = 10, start = 0, callback = [], backward_rounds = 1, **kwargs):
        """
        Args:
            loader (torch.DataLoader): training data loader
            epoch (int): number of epoch for training loop
            callback (list[Callback]): callable function will be called every epoch
                - parameters of callback
                    model (nn.Module): the training model
                    history (History): history of total log records
                    epoch (int): current epoch number
                    trainer (Trainer): self trainer
            start (int): epoch start from n round
            backward_rounds (int): backward after every n rounds 
        
        Returns:
            Module: the model with best performents
        """
        if loader is not None:
            self.state.loader = loader
        
        if self.state.loader is None:
            raise ValueError("loader is not set, please set a loader for trainning!")

        if not isinstance(callback, list):
            callback = [callback]
        
        # setup callbacks
        for c in callback:
            self.event.register("epoch:end", c)

        # distrubution trainning
        if self.state.distributor is not None:
            self.state.distributor.spawn(train_loop, self, **kwargs)
            # distribute_trainer.shutdown()
        else:
            train_loop(self, loader = self.state.loader, epoch = epoch, start = start, backward_rounds = backward_rounds)
        
        return self.state.module
    

    @torch.no_grad()
    def evaluate(self, loader, callback = None):
        """evalute model

        Args:
            loader (torch.DataLoader): evaluation data loader
            callback (callable): callback function
        """
        if callback and not isinstance(callback, Callback):
            callback = Callback(callback)
        
        # init progress bar
        p = Progress(loader)
        p.prefix = f"Evaluate"

        history = History()

        model = self.state.module
        step = self.state.step

        model.eval()

        history.start()
        
        loss = 0.
        for i, batch in enumerate(p, start = 1):
            # step fit
            if self.loss is None:
                l = step(model, batch)
            else:
                l = step(model, batch, loss=self.loss)

            # log loss
            history.log('loss', l)

            loss += (l.item() - loss) / i
            p.suffix = 'loss:{:.4f}'.format(loss)
        
        history.end()
        
        if callable(callback):
            callback(
                epoch = None,
                history = history,
                trainer = self,
                model = model,
            )
        
        return history



def train_loop(trainer, loader = None, epoch = 10, start = 0, backward_rounds = 1):
    # init progress bar
    p = Progress(loader)


    model = trainer.module
    loader = loader or trainer.loader
    step = trainer.state.step
    optimizer = trainer.optimizer
    
    for ep in range(start, epoch):
        # set model to train mode
        model.train()

        p.prefix = f"Epoch:{ep}"

        # setup a new history for model in each epoch
        history = History()
        trainer.histories.append(history)

        # setup callback params
        params = {
            "model": model,
            "history": history,
            "epoch": ep,
            "trainer": trainer,
            "progress": p,
        }

        trainer.event.emit("epoch:start", **params)
        
        # start of history
        history.start()

        loss = 0.
        backward_loss = 0.
        for i, batch in enumerate(p, start = 1):
            trainer.event.emit("batch:start", batch = batch, **params)

            # step fit
            if trainer.loss is None:
                l = step(model, batch)
            else:
                l = step(model, batch, loss=trainer.loss)

            # log loss
            history.log('loss', l)

            backward_loss = l + backward_loss
            if i % backward_rounds == 0 or i == len(p):
                optimizer.zero_grad()
                backward_loss.backward()
                optimizer.step()
                
                # reset backward loss
                backward_loss = 0.
            
            loss += (l.item() - loss) / i
            p.suffix = 'loss:{:.4f}'.format(loss)

            trainer.event.emit("batch:end", batch = batch, **params)

        # END of history
        history.end()

        with torch.no_grad():
            trainer.event.emit("epoch:end", **params)
            trainer.event.emit("earlystop:check", **params)
        
        # check if trainer need terminate
        if trainer.status == TrainerStatus.TERMINATED:
            break
