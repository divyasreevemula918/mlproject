import sys
import os
from dataclasses import dataclass
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from src.utils import save_object,evaluate_model
@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("splitting dependent and independent variables from train test array")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "CatBoost":CatBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "Linear Regressor":LinearRegression()

            }
            params = {
                "Random Forest":{
                    'n_estimators':[50,100],
                    'max_depth':[5,10]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01],
                    'n_estimators':[50,100]
                },
                "AdaBoost":{
                    'learning_rate':[0.1],
                    'n_estimators':[50]
                },
                "CatBoost":{
                    'depth':[6],
                    'learning_rate':[0.1],
                    'iterations':[50]
                },
                "XGBRegressor":{
                    'learning_rate':[0.1],
                    'n_estimators':[50],
                    'max_depth':[5]
                },
                "Decision Tree":{
                    'max_depth':[5,10]
                },
                "KNN":{
                    'n_neighbors':[5,7]
                },
                "Linear Regressor":{
                    'fit_intercept':[True,False]
                }
            }
                
            model_report,trained_models=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = trained_models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"best model found on both traing and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            print(r2_square)
            return {
                "model_name": best_model_name,
                "r2_score": r2_square,
                "best_cv_score": best_model_score
            }

            
           
        except Exception as e:
            raise CustomException(e,sys)
                                         