from src.config.configuration import Configuration
from src.components.stages.training import Training


class TrainingPipeline():
    def __init__(self):

        configuration = Configuration()
        self.training_config = configuration.training_config()
        self.data_transformation_config=configuration.data_transformation_config()

    def run_training(self):

        training = Training(self.training_config,self.data_transformation_config)
        training_loss,test_loss=training.initiate_training()


if __name__=="__main__":
    
    training_pipeline=TrainingPipeline()
    training_pipeline.run_training()
