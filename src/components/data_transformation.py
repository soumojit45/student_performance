import os 
import sys 
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numeric_cols = ['writing_score','reading_score']
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Categorical columns : {cat_cols}')
            logging.info(f'Numerical columns : {numeric_cols}')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numeric_cols),
                    ('cat_pipeline',cat_pipeline,cat_cols)
                ]
            )

            return preprocessor
        
        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj()
            target_col_name = "math_score"
            numeric_cols = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info(" Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]

            logging.info(f"saved preprocessing object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)