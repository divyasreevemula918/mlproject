print("FILE STARTED")

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

print("IMPORTS DONE")

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        print("CLASS INITIALIZED ")

    def initiate_data_ingestion(self):
        print("FUNCTION CALLED")

        df = pd.read_csv('notebook/data/stud.csv')
        print("CSV READ")

        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
        print("ARTIFACTS FOLDER READY")

        df.to_csv(self.ingestion_config.raw_data_path, index=False)
        print("RAW DATA SAVED ")

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        print("DATA SPLIT")

        train_set.to_csv(self.ingestion_config.train_data_path, index=False)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False)
        print("TRAIN & TEST SAVED")

        print("INGESTION COMPLETED")


if __name__ == "__main__":
    print("MAIN BLOCK ENTERED")
    obj = DataIngestion()
    obj.initiate_data_ingestion()
