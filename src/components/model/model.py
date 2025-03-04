import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from src.entity.config_entity import ModelConfig
from src.utils import save_obj
from src.exception.exception import ExceptionNetwork, sys

class ModelIngestion():
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def Loading_Model(self):
        try:
            model = resnet34(ResNet34_Weights.DEFAULT)
            
            for layers in model.parameters():
                layers.requires_grad = False
            
            return model
        except Exception as e:
            raise ExceptionNetwork(e, sys)
            
    def Change_Last_Layer(self, model, layer_size):
        try:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, layer_size)

            return model
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def ingestion_model(self):
        try:
            model = self.Loading_Model()
            model = self.Change_Last_Layer(model=model, layer_size=self.config.label_size)
            save_obj(model, save_path=self.config.model_save_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)
