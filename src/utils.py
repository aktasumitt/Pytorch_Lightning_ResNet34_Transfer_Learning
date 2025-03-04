from lightning.pytorch.callbacks import ModelCheckpoint
import json
import os
import yaml
import torch
from box import ConfigBox
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys

def save_as_json(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path,"w") as f:
            json.dump(data,f)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)

def load_json(path:Path):
    try:
        path=Path(path)
        with open(path, "r") as f:
            loaded_data = json.load(f)
        
        return ConfigBox(loaded_data)
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_as_yaml(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, allow_unicode=True)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_yaml(path:Path):  
    try:     
        path=Path(path)
        with open(f"{path}", "r", encoding="utf-8") as file:
            loaded_dict = yaml.safe_load(file) 
        
        return ConfigBox(loaded_dict)
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_obj(data,save_path:Path):
    try:    
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        torch.save(data,f=save_path)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)
        
        
def load_obj(path:Path):
    try:    
        path=Path(path)
        obj=torch.load(f=path,map_location=torch.device("cpu"),weights_only=False)
            
        return obj
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def Save_MyCheckpoint(save_dir):
    checkpoint=ModelCheckpoint(dirpath=save_dir,
                               filename="{epoch}-{steps}",
                               monitor="val_loss",
                               mode="min",
                               save_top_k=1,
                               save_last=True,
                               save_on_train_epoch_end=True)
    return checkpoint

    

    