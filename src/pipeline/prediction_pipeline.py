from src.config.configuration import Configuration
from src.components.stages.predictions.prediction import Prediction

class PredictionPipeline():
    def __init__(self):

        configuration=Configuration()
        self.prediction_config=configuration.prediction_config()
    
    def run_prediction_pipeline(self):
    
        prediction=Prediction(self.prediction_config)
        predict_results=prediction.convert_to_label_name()
        
        return predict_results
        
if __name__=="__main__":
    prediction_pipeline=PredictionPipeline()
    predict_results=prediction_pipeline.run_prediction_pipeline()