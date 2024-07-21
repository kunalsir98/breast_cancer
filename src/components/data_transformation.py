import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical features
            numerical_cols = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst',
                'symmetry_worst', 'fractal_dimension_worst'
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Trim whitespace from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Log the columns of the dataframes
            logging.info(f"Train DataFrame columns: {train_df.columns}")
            logging.info(f"Test DataFrame columns: {test_df.columns}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "diagnosis"

            # Check if target column exists in both dataframes
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in train or test DataFrame", sys)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Encode target variables
            target_feature_train_arr = target_feature_train_df.map({'M': 1, 'B': 0}).values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.map({'M': 1, 'B': 0}).values.reshape(-1, 1)

            # Concatenate input features with target features
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
