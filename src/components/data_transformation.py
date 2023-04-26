import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preproccesor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            categories_col = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order','Type_of_vehicle', 'Festival', 'City']
            numerical_col = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 'Order_pickup_time', 'Year', 'Month', 'Day']
            
            logging.info('Pipeline Initiated')
            ## Numerical Pipeline
            num_pipeline=Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('Standarize',StandardScaler()),])

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('OneHotencoder',OneHotEncoder(drop='first',sparse=False)),
                        ('Standarize',StandardScaler()),]
                )
            logging.info(f'Categorical Columns :- {categories_col}')
            logging.info(f'Numerical Columns :-  {numerical_col}')
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_col),
                ('cat_pipeline',cat_pipeline,categories_col)
            ])
            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
        
    def initate_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and Test data Completed')

            logging.info('Obtaining Preprocessing Object')
            preprocessing_obj = self.get_transformation_object()

            target_column_name = 'Time_taken (min)'
            # numerical_col = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 'Order_pickup_time', 'Year', 'Month', 'Day']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(" Appling Preprocessing obj on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)           
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]  
            
            logging.info("Saving Preprocessing Object.")
            save_object(file_path=self.data_transformation_config.preproccesor_obj_file,obj=preprocessing_obj)
            logging.info("Preprocessor Saved as Pickle")
            return (train_arr,test_arr,self.data_transformation_config.preproccesor_obj_file)

        except Exception as e:
            raise CustomException(e,sys)    