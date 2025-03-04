from src.config.configuration import Configuration
from src.components.stages.testing import Testing


class TestPipeline():
    def __init__(self):

        configuration = Configuration()
        self.test_config = configuration.test_config()
        self.data_config = configuration.data_transformation_config()

    def run_testing(self):

        testing = Testing(self.test_config,self.data_config)
        test_result=testing.initiate_test()
        return test_result


if __name__=="__main__":
    
    test_pipeline=TestPipeline()
    test_result=test_pipeline.run_testing()
    


