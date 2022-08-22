import torch
import numpy as np
from torch import optim

from .history import History
from .callback import callback as Callback
from .event import Event

from ...utils.progress import Progress

DISTRIBUTED_MODE = "distributed"
STANDALONE_MODE = "standalone"

TRAINER_INIT = "init"
TRAINER_RUNNING = "running"
TRAINER_TERMINATED = "terminated"

class Trainer(Event):
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
        super().__init__()

        self.set_model(model)

        self.loader = loader

        self._mode = STANDALONE_MODE
        self._state = TRAINER_INIT
        
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
        
        self.optimizer = optimizer

        self.loss = loss

        # set default early stopping
        if early_stopping is None:
            from .earlystop import loss_stopping
            early_stopping = loss_stopping()
        
        # self.early_stop = early_stopping
        self.register("epoch:end", early_stopping)

        from collections import deque
        self.history = deque(maxlen = keep_history)
    

    @property
    def state(self):
        return self._state
    

    def terminate(self):
        self._state = TRAINER_TERMINATED
    

    def run(self):
        self._state = TRAINER_RUNNING
    

    def set_model(self, model):
        """setup model
        """
        from ..module import Module
        
        if isinstance(model, Module):
            self.fit_step(model.__class__.fit_step)
        
        self.model = model
    

    def fit_step(self, func):
        self._step = func
        
        return func


    # initialize enviroment setting
    def distributed(self, address, workers = 4, gpu = False):
        '''setting distribution enviroment and initial a ray cluster connection

        Args: 
            address (string): the head of ray cluster address
            workers (int): compute task's resource
            gpu (Booleans): whether use GPU, "True" or "False"
        '''
        self._mode = DISTRIBUTED_MODE
        self._workers = workers
        self._gpu = gpu
        
        import ray
        if not ray.is_initialized():
            ray.init(address)
    

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
        for idx, c in enumerate(callback):
            if not isinstance(c, Callback):
                callback[idx] = Callback(c)

        loader = self.loader
        model = self.model

        if self._mode == DISTRIBUTED_MODE:
            import ray.train as train
            # TODO prepare loader and model
            loader = train.torch.prepare_data_loader(loader)
            model = train.torch.prepare_model(model)
            # TODO: remove this patch for dist
            model.fit_step = self.model.fit_step
            model.state_dict = self.model.state_dict
            model.log = self.model.log
       
        # init progress bar
        p = Progress(loader)
        
        for ep in range(start, epoch):
            # set model to train mode
            model.train()

            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            history = History()
            self.history.append(history)

            # setup callback params
            callback_params = {
                "model": model,
                "history": history,
                "epoch": ep,
                "trainer": self,
                "progress": p,
            }

            self.emit("epoch:start", **callback_params)
            
            # start of history
            history.start()

            loss = 0.
            backward_loss = 0.
            for i, batch in enumerate(p, start = 1):
                self.emit("batch:start", batch = batch, **callback_params)

                # step fit
                if self.loss is None:
                    l = self._step(model, batch)
                else:
                    l = self._step(model, batch, loss=self.loss)

                # log loss
                history.log('loss', l)
                backward_loss = l + backward_loss
                if i % backward_rounds == 0 or i == len(p):
                    self.optimizer.zero_grad()
                    backward_loss.backward()
                    self.optimizer.step()
                    
                    # reset backward loss
                    backward_loss = 0.
                
                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)

                self.emit("batch:end", batch = batch, **callback_params)

            # END of history
            history.end()

            with torch.no_grad():
                if isinstance(callback, list):
                    for hook in callback:
                        hook(**callback_params)
                
                self.emit("epoch:end", **callback_params)
            
            # check if trainer need terminate
            if self.state == TRAINER_TERMINATED:
                break
        

    def train(self, loader = None, epoch = 10, **kwargs):
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
            self.loader = loader
        
        if self.loader is None:
            raise ValueError("loader is not set, please set a loader for trainning!")
        
        config = {
            "epoch": epoch,
            **kwargs,
        }

        # distrubution trainning
        if self._mode == DISTRIBUTED_MODE:
            from ray.train.trainer import Trainer
            distribute_trainer = Trainer(
                backend = "torch",
                num_workers = self._workers,
                use_gpu = self._gpu,
            )
            distribute_trainer.start()
            result = distribute_trainer.run(
                train_func = self._train,
                config = config  
            )
            print(result)
            distribute_trainer.shutdown()
        else:
            self._train(config = config) 
        
        return self.model
    

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
        self.model._history = history

        self.model.eval()
        
        loss = 0.
        for i, batch in enumerate(p, start = 1):
            # step fit
            if self.loss is None:
                l = self.model.fit_step(batch)
            else:
                l = self.model.fit_step(batch, loss=self.loss)

            # log loss
            self.model.log('loss', l)

            loss += (l.item() - loss) / i
            p.suffix = 'loss:{:.4f}'.format(loss)
        
        if callable(callback):
            callback(
                epoch = None,
                history = history,
                trainer = self,
                model = self.model,
            )
        
        return history
