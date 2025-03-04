from lightning.pytorch import Trainer
from src.entity.config_entity import TestConfig,DataTransformationConfig
from src.components.stages.lightning_module import LightningTrainigModule
from src.components.data_transformation.data_transformation import LightDataTransformation
from src.utils import load_obj
from src.exception.exception import ExceptionNetwork, sys

class Testing():
    def __init__(self,config_test:TestConfig,config_data_transformation:DataTransformationConfig):
        try:
            self.config_test=config_test
            self.config_data_transformation=config_data_transformation
            self.model=load_obj(self.config_test.test_model_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)
        
    def create_data_module(self):
        try:
            data_module=LightDataTransformation(self.config_data_transformation)
            return data_module
        except Exception as e:
            raise ExceptionNetwork(e, sys)
        
    def load_checkpoints(self):
        try:
            test_module=LightningTrainigModule.load_from_checkpoint(checkpoint_path=self.config_test.best_checkpoints_path+"\\last-v1.ckpt",
                                                                    model=self.model,
                                                                    label_size=self.config_test.label_size)
            return test_module
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def create_tester(self):
        try:
            # Trainer
            tester=Trainer(devices=1,
                            accelerator="cpu")
            return tester
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def initiate_test(self):
        try:
            data_module=self.create_data_module()
            test_module=self.load_checkpoints()
            tester=self.create_tester()
            
            test_loss = tester.test(model=test_module,datamodule=data_module)
            
            return test_loss
        except Exception as e:
            raise ExceptionNetwork(e, sys)
