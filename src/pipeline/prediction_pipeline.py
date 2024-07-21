import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load the preprocessor and model objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the loaded model
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 radius_mean: float,
                 texture_mean: float,
                 perimeter_mean: float,
                 area_mean: float,
                 smoothness_mean: float,
                 compactness_mean: float,
                 concavity_mean: float,
                 concave_points_mean: float,
                 symmetry_mean: float,
                 fractal_dimension_mean: float,
                 radius_se: float,
                 texture_se: float,
                 perimeter_se: float,
                 area_se: float,
                 smoothness_se: float,
                 compactness_se: float,
                 concavity_se: float,
                 concave_points_se: float,
                 symmetry_se: float,
                 fractal_dimension_se: float,
                 radius_worst: float,
                 texture_worst: float,
                 perimeter_worst: float,
                 area_worst: float,
                 smoothness_worst: float,
                 compactness_worst: float,
                 concavity_worst: float,
                 concave_points_worst: float,
                 symmetry_worst: float,
                 fractal_dimension_worst: float):
        
        self.radius_mean = radius_mean
        self.texture_mean = texture_mean
        self.perimeter_mean = perimeter_mean
        self.area_mean = area_mean
        self.smoothness_mean = smoothness_mean
        self.compactness_mean = compactness_mean
        self.concavity_mean = concavity_mean
        self.concave_points_mean = concave_points_mean
        self.symmetry_mean = symmetry_mean
        self.fractal_dimension_mean = fractal_dimension_mean
        self.radius_se = radius_se
        self.texture_se = texture_se
        self.perimeter_se = perimeter_se
        self.area_se = area_se
        self.smoothness_se = smoothness_se
        self.compactness_se = compactness_se
        self.concavity_se = concavity_se
        self.concave_points_se = concave_points_se
        self.symmetry_se = symmetry_se
        self.fractal_dimension_se = fractal_dimension_se
        self.radius_worst = radius_worst
        self.texture_worst = texture_worst
        self.perimeter_worst = perimeter_worst
        self.area_worst = area_worst
        self.smoothness_worst = smoothness_worst
        self.compactness_worst = compactness_worst
        self.concavity_worst = concavity_worst
        self.concave_points_worst = concave_points_worst
        self.symmetry_worst = symmetry_worst
        self.fractal_dimension_worst = fractal_dimension_worst

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'radius_mean': [self.radius_mean],
                'texture_mean': [self.texture_mean],
                'perimeter_mean': [self.perimeter_mean],
                'area_mean': [self.area_mean],
                'smoothness_mean': [self.smoothness_mean],
                'compactness_mean': [self.compactness_mean],
                'concavity_mean': [self.concavity_mean],
                'concave points_mean': [self.concave_points_mean],
                'symmetry_mean': [self.symmetry_mean],
                'fractal_dimension_mean': [self.fractal_dimension_mean],
                'radius_se': [self.radius_se],
                'texture_se': [self.texture_se],
                'perimeter_se': [self.perimeter_se],
                'area_se': [self.area_se],
                'smoothness_se': [self.smoothness_se],
                'compactness_se': [self.compactness_se],
                'concavity_se': [self.concavity_se],
                'concave points_se': [self.concave_points_se],
                'symmetry_se': [self.symmetry_se],
                'fractal_dimension_se': [self.fractal_dimension_se],
                'radius_worst': [self.radius_worst],
                'texture_worst': [self.texture_worst],
                'perimeter_worst': [self.perimeter_worst],
                'area_worst': [self.area_worst],
                'smoothness_worst': [self.smoothness_worst],
                'compactness_worst': [self.compactness_worst],
                'concavity_worst': [self.concavity_worst],
                'concave points_worst': [self.concave_points_worst],
                'symmetry_worst': [self.symmetry_worst],
                'fractal_dimension_worst': [self.fractal_dimension_worst]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
