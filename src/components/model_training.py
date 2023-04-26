import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys
import os
from sklearn.metrics import r2_score

@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initate_model_trainer(self,train_arr,test_arr):
        logging.info("Model Train Intiated")
        try:
            logging.info("Split Train and test Input data ")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Models 
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(alpha=0.05,max_iter=10000),
                'Ridge':Ridge(max_iter=1000,solver='svd'),
                'Elasticnet':ElasticNet(alpha=0.05,max_iter=5000),
                'RFR': RandomForestRegressor(max_depth=10, n_estimators=1000, random_state=123,verbose=0)
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models)
            #  Max R2 
            BEST_Model_SCORE  = max(sorted(model_report.values())) 

            # Best R2 Index Get
            BEST_model_name = list(model_report.keys())[list(model_report.values()).index(BEST_Model_SCORE)]

            best_model = models[BEST_model_name]

            if BEST_Model_SCORE < 0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best Model Found on both Training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info('Best Model Saved as Pickle') 

            # R2
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square   

        except Exception as e :
            raise CustomException(e,sys)
