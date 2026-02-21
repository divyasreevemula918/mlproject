print("TRAIN PIPELINE FILE RUNNING ")

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

print("IMPORTS DONE ")

data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()

print("INGESTION DONE ")

data_transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

print("TRANSFORMATION DONE ")

model_trainer = ModelTrainer()
print("MODEL TRAINER CREATED ")

result = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

print("FINAL MODEL RESULT ")
print(result)