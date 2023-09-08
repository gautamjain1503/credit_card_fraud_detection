import pandas as pd
from credit_card_fraud_detection.components.pre_processer import DataTransformation
from credit_card_fraud_detection.entity.config_entity import DataPreprocesserConfig
from credit_card_fraud_detection.utils.common import load_bin
from pathlib import Path

class PredictPipeline:
    def __init__(self,config: DataPreprocesserConfig, model_path: Path):
        self.data_transformation_config=config
        self.model_path=model_path
        self.preprocessor_obj=load_bin(self.data_transformation_config.preprocessor_dir)

    def predict_fraud(self,df):
        df=self.preprocessor_obj.transform(df)
        model=load_bin(self.model_path)
        result=model.predict(df)
        return result[0]




class CustomData:
    def __init__(  self,
                 cc_num: int,
                 merchant: str,
                 category: str,
                 amt: float,
                 gender: str,
                 city: str,
                 state: str,
                 zip: int,
                 lat: float,
                 long: float,
                 city_pop: int,
                 job: str,
                 dob: str,
                 unix_time: int,
                 merch_lat: float,
                 merch_long: float):
        
        self.cc_num=cc_num
        self.merchant=merchant
        self.category=category
        self.amt=amt
        self.gender=gender
        self.city=city
        self.state=state
        self.zip=zip
        self.lat=lat
        self.long=long
        self.city_pop=city_pop
        self.job=job
        self.dob=dob
        self.unix_time=unix_time
        self.merch_lat=merch_lat
        self.merch_long=merch_long


    def get_data_as_data_frame(self):
        custom_data_input_dict = {
            "cc_num":[self.cc_num],
            "merchant":[self.merchant],
            "category":[self.category],
            "amt":[self.amt],
            "gender":[self.gender],
            "city":[self.city],
            "state":[self.state],
            "zip":[self.zip],
            "lat":[self.lat],
            "long":[self.long],
            "city_pop":[self.city_pop],
            "job":[self.job],
            "dob":[self.dob],
            "unix_time":[self.unix_time],
            "merch_lat":[self.merch_lat],
            "merch_long":[self.merch_long]
        }

        return pd.DataFrame(custom_data_input_dict)
