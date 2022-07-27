import torch
import numpy as np
from torch import optim

from .history import History
from .callback import callback as Callback

from ...utils.progress import Progress



class Trainer:
    def __init__(self, model, loader = None, optimizer = None, keep_history = None,
                early_stopping = None):
        """
        Args:
            model (nn.Module): model will be trained
            loader (torch.DataLoader): training data loader
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            keep_history (int): keep the last n-th epoch logs, `None` will keep all
        """
        self.model = model
        self.loader = loader
        self._distribute_trainer=None # distrubution trainer initialization
        self._distrubution_train=False
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
        
        self.optimizer = optimizer

        # set default early stopping
        if early_stopping is None:
            from .earlystop import loss_scoring
            early_stopping = loss_scoring
        
        self.early_stop = early_stopping

        from collections import deque
        self.history = deque(maxlen = keep_history)
    
    # initialize enviroment setting
    def distributed(self,num_works=4,use_gpu=False):
        '''
        
        Args: 




        '''
        import ray
        from ray.train.trainer import Trainer
        from ray.train.callbacks import JsonLoggerCallback
        self._distrubution_train=True
        self.num_works=num_works
        self.use_gpu=use_gpu
        if not ray.is_initialized():
            ray.init('ray://172.20.159.144:10001')  # ray server address
        self._distribute_trainer = Trainer(backend="torch", num_workers=self.num_workers, use_gpu=self.use_gpu)
        

    # def _train(self,lr,batch_size,epoches):
    #     import ray.train as train
    #     loss_fn = torch.nn.CrossEntropyLoss()
    #     # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #     p=Progress(self.loader)
    #     loss_results = []
    #     for _ in range(epoches):
    #         size = len(datatrain_loader) // train.world_size()
    #         model.train()
    #         p.prefix = f"Epoch:{_}"
    #         history = History()
    #         self.history.append(history)
    #         self.model._history = history
    #         for batch, (X, y) in enumerate(datatrain_loader):
    #             # Compute prediction error
    #             # pred = model(X)
    #             pred=self.model.fit_step(X)
    #             loss = loss_fn(pred, y)
    #             self.model.log('loss', loss)
    #             # Backpropagation
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             if batch % 100 == 0:
    #                 loss, current = loss.item(), batch * len(X)
    #                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    #     train.report(loss=loss)
    #     loss_results.append(loss)


    def _train(self, epoch, callback = None, start=0, backward_rounds = 1):

        if self._distrubution_train:
            import ray.train as train
            loader = train.torch.prepare_data_loader(self.loader)
            model = train.torch.prepare_model(self.model)
        else:
            loader = self.loader
            model = self.model
        
        # init progress bar
        p = Progress(loader)
        for ep in range(start, epoch):
            # set model to train mode
            model.train()

            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            history = History()
            self.history.append(history)
            model._history = history

            loss = 0.
            backward_loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                l = model.fit_step(batch)

                # log loss
                model.log('loss', l)
                
                backward_loss = l + backward_loss
                if i % backward_rounds == 0 or i == len(p):
                    self.optimizer.zero_grad()
                    backward_loss.backward()
                    self.optimizer.step()
                    
                    # reset backward loss
                    backward_loss = 0.

                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)

            # setup callback params
            callback_params = {
                "model": model,
                "history": history,
                "epoch": ep,
                "trainer": self,
            }

            with torch.no_grad():
                if self.early_stop and self.early_stop(**callback_params):
                    # set best state to model
                    best_state = self.early_stop.get_best_state()
                    model.load_state_dict(best_state)
                    break
                
                if callable(callback):
                    callback(**callback_params)





    def train(self, loader = None, epoch = 10, callback = None, **kwargs):
        """
        Args:
            loader (torch.DataLoader): training data loader
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
            start (int): epoch start from n round
            backward_rounds (int): backward after every n rounds 
        """
        if loader is not None:
            self.loader = loader
        
        if self.loader is None:
            raise ValueError("loader is not set, please set a loader for trainning!")

        if callback and not isinstance(callback, Callback):
            callback = Callback(callback)

        # distrubution trainning
        if self._distrubution:
            self._distribute_trainer.start()
            result = self._distribute_trainer.run(
                train_func=self._train,
                config={"epoch": epoch,**kwargs},
                # callbacks=[JsonLoggerCallback()],  
            )
            print(result)
            self._distribute_trainer.shutdown()
        else:
            self._train(self, epoch, callback = None, start=0, backward_rounds = 1) 
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
            l = self.model.fit_step(batch)

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