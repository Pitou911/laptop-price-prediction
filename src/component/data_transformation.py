import sys
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["Inches", "Ram", "Weight", "CPU Clock Speed", "Width", "Height"]
            categorical_columns = [
                "Gpu",
                "OpSys",
                "CPU Manufacturer",
                "CPU Series",
                "CPU Model",
                "Capacity",
                "MemoryType",
                "Laptop"
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            pass
        except:
            pass