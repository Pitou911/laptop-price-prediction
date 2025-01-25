from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import sys
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_model(self):
        try:
            # Split data into features and target
            X = self.dataframe.drop(columns=['price'])
            y = self.dataframe['price']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize RandomForestRegressor
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            # Predict on test data
            y_pred = model.predict(X_test)
            
            # Evaluate model
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logging.info(f"Model R2 Score: {r2}")
            logging.info(f"Model Mean Squared Error: {mse}")
            
            return model
        except Exception as e:
            raise CustomException(e,sys)
