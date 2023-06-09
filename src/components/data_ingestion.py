from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_training import ModeltrainerConfig,ModelTrainer
from src.components.data_transformation import DataTransformation,DataTransformationConfig 

@dataclass
class DataIngestionConfig:
    train_data_path  = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_ingestion(self):
        logging.info('Data Ingestion started')
        try:
            # Reading Final Data -- Zomato Time Taken Dataset  
            df=pd.read_csv(os.path.join('notebooks/data','clean_data.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__  ==  "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_ingestion()
    data_tranformation = DataTransformation()
    train_arr,test_arr,_ = data_tranformation.initate_transformation(train_path,test_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initate_model_trainer(train_arr,test_arr))