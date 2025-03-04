from src.components.data_transformation.dataset import CreateDataset
from src.utils import save_obj
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import DataTransformationConfig
from src.logging.logger import logger
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from src.components.data_transformation.transformer import get_transformer,get_img_label_dict

# Lightning Data Module
class LightDataTransformation(LightningDataModule):
    
    def __init__(self,config:DataTransformationConfig):
        super(LightDataTransformation,self).__init__()
        try:
            self.config=config
            self.train_transformer,self.valid_transformer,self.test_transformer=get_transformer(self.config.img_resize_size)
            self.image_label_dict_train=get_img_label_dict(self.config.train_data_path,self.config.labels)
            self.image_label_dict_valid=get_img_label_dict(self.config.valid_data_path,self.config.labels)
            self.image_label_dict_test=get_img_label_dict(self.config.test_data_path,self.config.labels)
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    
    def setup(self, stage):
        try:
            
            if stage=="fit":
                self.train_dataset=CreateDataset(img_label_dict=self.image_label_dict_train,transformer=self.train_transformer)
                
                self.valid_dataset=CreateDataset(img_label_dict=self.image_label_dict_valid,transformer=self.valid_transformer)
            
            if stage=="test":
                self.test_dataset=CreateDataset(img_label_dict=self.image_label_dict_test,transformer=self.test_transformer)
                
        except Exception as e:
            raise ExceptionNetwork(e,sys)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.config.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,batch_size=self.config.batch_size,shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.config.batch_size,shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.config.batch_size,shuffle=True)
                