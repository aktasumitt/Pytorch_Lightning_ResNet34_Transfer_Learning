from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from src.utils import Save_MyCheckpoint
from src.entity.config_entity import TrainingConfig,DataTransformationConfig
from src.components.stages.lightning_module import LightningTrainigModule
from src.components.data_transformation.data_transformation import LightDataTransformation
from src.utils import load_obj

import dagshub
dagshub.init(repo_owner='umitaktas', repo_name='Pytorch_Lightning_ResNet34_Transfer_Learning', mlflow=True)


class Training():
    def __init__(self,config_train:TrainingConfig,config_data_transformation:DataTransformationConfig):
        self.config_train=config_train
        self.config_data_transformation=config_data_transformation
        self.model=load_obj(self.config_train.model_path)

    def create_data_module(self):
        data_module=LightDataTransformation(self.config_data_transformation)
        return data_module
        
    
    def load_module_and_checkpoints(self,load_checkpoints:bool=False):

        if load_checkpoints==True:
            training_module=LightningTrainigModule.load_from_checkpoint(checkpoint_path=self.config_train.checkpoint_path+"\\last.ckpt",
                                                                        model=self.model,
                                                                        label_size=self.config_train.labe_size,
                                                                        learnin_rate=self.config_train.learning_rate)
        else:
            training_module=LightningTrainigModule(label_size=self.config_train.labe_size,
                                                    learnin_rate=self.config_train.learning_rate,
                                                    model=self.model)
            
        return training_module
    
    def create_logger(self):
           
        # mlflow logger  
        logger=MLFlowLogger(experiment_name="Resnet34_Animals_Classification",
                            tracking_uri="https://dagshub.com/umitaktas/Pytorch_Lightning_ResNet34_Transfer_Learning.mlflow")
        return logger


    def create_trainer(self,logger):
        # Trainer
        trainer=Trainer(max_epochs=self.config_train.epochs,
                        devices=1,
                        accelerator="gpu",
                        logger=logger,
                        callbacks=[Save_MyCheckpoint(save_dir=self.config_train.checkpoint_path)]
                        )
        return trainer

    def initiate_training(self):
        data_module=self.create_data_module()
        
        training_module=self.load_module_and_checkpoints(self.config_train.load_checkpoint)
        logger=self.create_logger()
        trainer=self.create_trainer(logger)
        
        train_loss = trainer.fit(model=training_module,datamodule=data_module)
        test_loss = trainer.test(model=training_module,datamodule=data_module)
        
        return train_loss,test_loss

            