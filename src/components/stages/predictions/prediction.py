from lightning.pytorch import Trainer
from src.entity.config_entity import PredictionConfig
from src.components.stages.lightning_module import LightningTrainigModule
from src.utils import load_obj
from src.components.stages.predictions.load_data import load_Data_to_transform, transform_to_dataloader
from src.exception.exception import ExceptionNetwork, sys

class Prediction():
    def __init__(self, config: PredictionConfig):
        try:
            self.config = config
            self.model = load_obj(self.config.final_model_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def load_data(self):
        try:
            image_label_dict = load_Data_to_transform(image_paths=self.config.predict_data_path)
            dataloader = transform_to_dataloader(image_label_dict, resize_size=self.config.image_size)
            return dataloader
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def load_checkpoints(self):
        try:
            test_module = LightningTrainigModule.load_from_checkpoint(checkpoint_path=self.config.checkpoint_path+"\\last-v1.ckpt",
                                                                      model=self.model,
                                                                      label_size=self.config.label_size)
            return test_module
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def create_tester(self):
        try:
            # Trainer
            tester = Trainer(devices=1, accelerator="cpu")
            return tester
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def initiate_prediction(self):
        try:
            dataloader = self.load_data()
            test_module = self.load_checkpoints()
            tester = self.create_tester()
            
            prediction_result = tester.predict(model=test_module, dataloaders=dataloader)
            print(prediction_result)
            print(type(prediction_result))
            
            return prediction_result[0]["predict_labels"]
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def convert_to_label_name(self):
        try:
            predicted_label_names = []
            prediction_results = self.initiate_prediction()
            labels_converted = {v: k for k, v in self.config.labels.items()}
            print(prediction_results)
            for predicted_label in prediction_results[0]:
                print(predicted_label)
                label_name = labels_converted[predicted_label.item()]
                predicted_label_names.append(label_name)
            
            print(predicted_label_names)
            return predicted_label_names
        except Exception as e:
            raise ExceptionNetwork(e, sys)