class Configs():
    
    # After data ingestion
    DATA_LOCATION_PATH = "D:/Python (New)/Datasets/Animals.zip"
    ALL_DATA_SAVE_PATH = "all_data"
    TRAIN_DATA_PATH = "local_data/train"
    VALID_DATA_PATH = "local_data/valid"
    TEST_DATA_PATH = "local_data/test"
    EXAMPLE_DATA_FOR_PYTEST = "test/images.zip" # bu path pytest için testte kullanılan datanın pathi

    # After creating model
    MODEL_SAVE_PATH = "callbacks/model/model.pth"

    # After training
    CHECKPOINT_SAVE_DIR = "callbacks/checkpoints/"
    
    # After Testing
    TESTED_MODEL_SAVE_PATH = "callbacks/tested_model/tested_best_model.pth"
    
    # After prediction
    SAVE_PREDICTION_RESULT_PATH = "predict_artifact/results/result.json"
    PREDICTION_DATA_PATH= "predict_artifact/images"
    
    