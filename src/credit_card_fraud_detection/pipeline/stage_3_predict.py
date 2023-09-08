from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.predict import PredictPipeline, CustomData
from credit_card_fraud_detection import logger
from pathlib import Path

STAGE_NAME = "Prediction Pipeline"

class Predict:
    def __init__(self):
        pass

    def main(self, dictionary):
        config = ConfigurationManager()
        preprocessor_config = config.get_preprocesser_config()
        model_path=Path("artifacts/training/model.pkl")
        data=CustomData(cc_num=dictionary["cc_num"],
                        merchant=dictionary["merchant"],
                        category=dictionary["category"],
                        amt=dictionary["amt"],
                        gender=dictionary["gender"],
                        city=dictionary["city"],
                        state=dictionary["state"],
                        zip=dictionary["zip"],
                        lat=dictionary["lat"],
                        long=dictionary["long"],
                        city_pop=dictionary["city_pop"],
                        job=dictionary["job"],
                        dob=dictionary["dob"],
                        unix_time=dictionary["unix_time"],
                        merch_lat=dictionary["merch_lat"],
                        merch_long=dictionary["merch_long"],
        )
        df=data.get_data_as_data_frame()
        model = PredictPipeline(config=preprocessor_config, model_path=model_path)
        result=model.predict_fraud(df=df)
        logger.info(f">>>>>>  {result}  <<<<<<\n\nx==========x")
        return result




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dictionary={}
        dictionary["cc_num"]=2703186189652095
        dictionary["merchant"]="fraud_Rippin, Kub and Mann"
        dictionary["category"]="misc_net"
        dictionary["amt"]=4.97
        dictionary["gender"]="F"
        dictionary["city"]="Moravian Falls"
        dictionary["state"]="NC"
        dictionary["zip"]=28654
        dictionary["lat"]=36.0788
        dictionary["long"]=-81.1781
        dictionary["city_pop"]=3495
        dictionary["job"]="Psychologist, counselling"
        dictionary["dob"]="1988-03-09"
        dictionary["unix_time"]=1325376018
        dictionary["merch_lat"]=36.011293
        dictionary["merch_long"]=-82.048315
        print(dictionary)

        obj = Predict()
        obj.main(dictionary=dictionary)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
