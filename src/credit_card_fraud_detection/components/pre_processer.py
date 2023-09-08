from dataclasses import dataclass
import pandas as pd
import os
from pathlib import Path
from credit_card_fraud_detection.utils.common import save_bin
from credit_card_fraud_detection.entity.config_entity import DataPreprocesserConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self,config: DataPreprocesserConfig):
        self.data_transformation_config=config
        
    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        numerical_columns = [
            "cc_num",
            "amt",
            "zip",
            "lat",
            "long",
            "city_pop",
            "unix_time"
        ]
        categorical_columns = [
            'merchant',
            'category',
            'gender',
            'city',
            'state',
            'job',
            'dob'
        ]

        num_pipeline= Pipeline(
            steps=[
            ("MinMaxScaler",MinMaxScaler())
            ]
        )

        cat_pipeline=Pipeline(
            steps=[
            ("OrdinalEncoder",OrdinalEncoder()),
            ("MinMaxScaler",MinMaxScaler())
            ]
        )

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
        )

        return preprocessor            
        
        
    def initiate_data_transformation(self):

        df = pd.read_csv(self.data_transformation_config.data_dir)
        df.drop(["Unnamed: 0"],axis=1, inplace=True)
        X = df.drop(['is_fraud', 'first', 'last', 'street', 'trans_num','trans_date_trans_time'], axis = 1)
        Y = df['is_fraud']
        preprocessing_obj=self.get_data_transformer_object()
        X=preprocessing_obj.fit_transform(X)
        nm_sampler = NearMiss()
        x_sampled, y_sampled = nm_sampler.fit_resample(X, Y)
        x_train, x_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size = 0.2, random_state = 2)
        save_bin(path=self.data_transformation_config.preprocessor_dir,
                data=preprocessing_obj
        )

        return x_train, x_test, y_train, y_test