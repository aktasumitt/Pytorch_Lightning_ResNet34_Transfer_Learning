import torch
from src.logging.logger import logger
from src.exception.exception import ExceptionNetwork, sys

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification.accuracy import Accuracy

class LightningTrainigModule(LightningModule):
    
    def __init__(self, model, label_size: int = None, learnin_rate: float = None):
        try:
            super(LightningTrainigModule, self).__init__()
            
            self.LR = learnin_rate  # lr for optimizer
            self.model = model
            self.accuracy = Accuracy("multiclass", num_classes=label_size)  # For Calculating Acc
            
            # For Prediction
            self.real_labels_list = []
            self.predict_labels_list = []
        except Exception as e:
            raise ExceptionNetwork(e, sys)
            
    def training_step(self, data):
        try:
            img, label = data
            out = self.forward(img)
            loss = nn.CrossEntropyLoss()(out, label)
            train_acc = self.accuracy(out, label)
            
            self.log_dict({"Train_acc": train_acc, "Train_Loss": loss}, prog_bar=True, on_epoch=True, on_step=False)
            
            return loss
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def validation_step(self, data):
        try:
            img, label = data
            out = self.forward(img)
            loss = nn.CrossEntropyLoss()(out, label)
            train_acc = self.accuracy(out, label)
            
            self.log_dict({"Val_acc": train_acc, "val_loss": loss}, prog_bar=True, on_epoch=True, on_step=False)
            
            return loss
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def test_step(self, data):
        try:
            img, label = data
            out = self.forward(img)
            loss = nn.CrossEntropyLoss()(out, label)
            train_acc = self.accuracy(out, label)
            
            self.log_dict({"Test_acc": train_acc, "Test_Loss": loss}, prog_bar=True, on_epoch=True, on_step=False)
            
            return loss
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def predict_step(self, data):
        try:
            img, label = data
            out = self.forward(img)
            _, pred = torch.max(out, 1)
            
            self.predict_labels_list.append(pred)
            self.real_labels_list.append(label)
            
            return {"real_labels": self.real_labels_list, "predict_labels": self.predict_labels_list}
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def configure_optimizers(self):
        try:
            return torch.optim.Adam(params=self.parameters(), lr=self.LR)
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def forward(self, data):
        try:
            out = self.model(data) 
            return out
        except Exception as e:
            raise ExceptionNetwork(e, sys)
