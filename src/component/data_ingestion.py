import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.component.data_transformation import DataTransformation
# from src.component.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def load_and_clean_data(self):
        try:
            df = pd.read_csv('notebook\data\laptop_price.csv')

            if 'laptop_ID' in df.columns:
                df = df.drop(columns=['laptop_ID'], axis=1)
            df = df.drop_duplicates()
            df['CPU Manufacturer'] = df['Cpu'].apply(lambda x: x.split()[0])
            df['CPU Series'] = df['Cpu'].apply(lambda x: " ".join(x.split()[:2]))
            df['CPU Model'] = df['Cpu'].apply(lambda x: re.search(r'\d+[A-Za-z]*', x).group() if re.search(r'\d+[A-Za-z]*', x) else None)
            df['CPU Clock Speed'] = df['Cpu'].apply(lambda x: float(re.search(r'[\d\.]+GHz', x).group()[:-3]) if re.search(r'[\d\.]+GHz', x) else None)
        
            # Extract screen resolution
            df['ResolutionValue'] = df['ScreenResolution'].apply(lambda x: re.search(r'\b\d+x\d+\b', x).group() if re.search(r'\b\d+x\d+\b', x) else None)

            # Convert RAM to int
            df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

            # Extract memory details
            df['Capacity'] = df['Memory'].str.extract(r'(\d+(?:\.\d+)?\s?\w+)')
            df['MemoryType'] = df['Memory'].str.extract(r'(SSD|HDD|Flash Storage|Hybrid)', expand=False)

            # Convert weight to float
            df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

            # Combine columns for a new feature
            df['Laptop'] = df['Company'] + df['TypeName'] + df['Product']

            # Split resolution into width and height
            df[['Width', 'Height']] = df['ResolutionValue'].str.split('x', expand=True).astype(int)

            # Drop unnecessary columns
            del_cols = ['Company', 'Product', 'TypeName', 'ScreenResolution', 'ResolutionValue', 'Cpu', 'Memory']
            df = df.drop(columns=del_cols)

            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self, df):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__=='__main__':
    obj=DataIngestion()
    df = obj.load_and_clean_data()
    obj.initiate_data_ingestion(df)

    # data_tranformation = DataTranformation()
