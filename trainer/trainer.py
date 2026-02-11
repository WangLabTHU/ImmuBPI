import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from utils import PerformanceEvaluator, DataSaver, setup_logging
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    import argparse
    import torch.nn as nn
    from torch.utils.data import Dataset




class Trainer():
    def __init__(
        self, 
        args: "argparse.Namespace",
        model: "nn.Module",
        train_dataset: "Dataset",
        valid_dataset: "Dataset",
        criterion: "nn.Module",
        collate_function: Optional[Callable[..., Any]] = None
    ) -> None:
        self.args= args
        self.init_logging()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.criterion = criterion
        self.collate_function = collate_function
        self.load_dataset()
        logging.debug(f"CUDA: {torch.cuda.is_available()}")
        self.init_utils() # MARK: utils for save the cur, avg, best loss and others mertric values in training
        self.define_optimizer()



    def load_dataset(self) -> None:

        self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.args.batch_size, num_workers = self.args.num_workers, shuffle = True, collate_fn = self.collate_function)
        self.val_dataloader = DataLoader(self.valid_dataset, batch_size = self.args.batch_size, num_workers = self.args.num_workers, shuffle = True, collate_fn = self.collate_function)


    def init_utils(self):
        if self.args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.save_root)

        self.train_loss = DataSaver()
        self.val_best_saver = DataSaver()
        self.train_evaluator = PerformanceEvaluator()
        self.val_evaluator = PerformanceEvaluator()


    def reset_utils(self):
        self.train_loss.reset()
        self.val_best_saver.reset()
        self.train_evaluator.reset()
        self.val_evaluator.reset()
       

    def define_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.args.lr)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)


    def init_logging(self):
        # create new folder to saving checkpoints and log for each training

        if self.args.save_remark:
            self.save_root = Path("./models_saved") / self.args.save_remark / ('Fold_'+str(self.args.train_dataset['args']['fold']))
        else:
            self.save_root = Path("./models_saved") / datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')
        
        self.save_root.mkdir(parents=True)
        setup_logging(self.save_root)
        # also save args into log
        with open(self.save_root / 'config.yaml', 'wt') as f:
            yaml.safe_dump(vars(self.args), f, indent=4)


    def train(self):
        logging.debug('-------------------------------------------------')
        logging.debug('-------------------------------------------------')
        logging.debug('Standard Training Started')
        logging.debug('-------------------------------------------------')
        logging.debug('-------------------------------------------------')
        logging.info(self.model)
       
        torch.save(self.model.state_dict(), self.save_root /  'random.pth')
            
        for self.epoch in range(self.args.epoch):
           

            self.reset_utils()
            logging.info(f'\nEpoch: {self.epoch}')
            self.model.train()

            for batch, data in enumerate(self.train_dataloader):
                if torch.cuda.is_available():
                    data = {k: v.cuda() if isinstance(v, torch.Tensor) \
                            else v \
                            for k, v in data.items()}

                
                self.optimizer.zero_grad()
                output = self.model(**data) # type: torch.Tensor
                # TODO refactor loss to call function by self.critertion(**xx)
                if 'weight' not in data:
                    loss = self.criterion(output, data["label"])
                else:
                    loss = self.criterion(output, data["label"], **{'weight': data['weight']})
                loss.backward()
                self.optimizer.step()

                self.train_evaluator.update(
                    data["label"].cpu().detach().numpy(),   # to prob
                    output.softmax(dim=-1)[:,1].cpu().detach().numpy()
                )
                self.train_loss.update(loss.item())
                if (batch % self.args.interval == 0):
                    self.check_gradient(self.epoch*len(self.train_dataloader) + batch)
                    logging.info(f'batch {batch} | training loss: {self.train_loss.avg: .3f}')

                if self.args.debug:
                    break
                            
            self.train_evaluator.cal_performance()
            logging.info(f'training loss: {self.train_loss.avg:.3f} | total accuracy {self.train_evaluator.accu:.3f}')      

            self.valid()


    def valid(self):
        self.model.eval()
        for batch, data in enumerate(self.val_dataloader):
            if torch.cuda.is_available():
                data = {k: v.cuda() if isinstance(v, torch.Tensor) \
                            else v \
                            for k, v in data.items()}

            output = self.model(**data)
            #loss = criterion(output, data["label"])

            self.val_evaluator.update(
                data["label"].cpu().detach().numpy(), 
                output.softmax(dim=-1)[:,1].cpu().detach().numpy()
            )

        self.val_evaluator.cal_performance()
        if self.args.save_model:
            if self.val_evaluator.auc > self.val_best_saver.max:
                torch.save(self.model.state_dict(), self.save_root /  'best_model.pth')
                #print('*'*10 + f'Save Best Model Epoch {self.epoch}' + '*'*10 )
                logging.info('*'*10 + f'Save Best Model Epoch {self.epoch}' + '*'*10 )

        # torch.save(self.model.state_dict(), self.save_root /  'best_model.pth')
        self.val_best_saver.update(self.val_evaluator.auc, self.epoch)

        logging.info(f'Validation Accuracy {self.val_evaluator.accu:.4f} | AUC {self.val_evaluator.auc:.4f} | PRAUC {self.val_evaluator.prauc:.4f}')
        # early stop when there is no improvement in AUC in `x` epoch
        #logging.debug(f"{self.val_accu.argmax}, {self.val_accu.max}")



        if self.epoch >= 50 and self.val_best_saver.argmax + self.args.decrease_epoch <= self.epoch:
            logging.info(f'{self.args.decrease_epoch} epoch val AUC stops increasing')
            logging.info(f'Best Epoch %d Best Valid Accu %.4f'%(self.val_best_saver.argmax, self.val_best_saver.max))
            exit(0)

    def check_gradient(self, num):
        if self.args.tensorboard:
            for name, p in self.model.named_parameters():
                if p.grad is None: # skip no grad(freezing) parameters
                    continue
                self.writer.add_histogram(f"grad_{name}", p.grad, num, bins="auto")