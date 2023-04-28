import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
class PredictPipeline:
    def __init__(self):
        pass
    def Predict(self,features):
        try :
            model_path = os.path.join('artifacts','model.pkl')    
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                Age :int,
                DeliveryPersonRating : int,
                VechileCondition : int, 
                MultipleDeliveries: int,
                Total_Ordering_PickupTime : int,
                Year : int ,
                Month : int, 
                Day : int ,
                WeatherCondtion : str,
                RoadTrafficCondition : str,                 
                Typeofvechile : str,
                Typeoforder : str,
                Festival : str,
                City : str
                ):
        self.Age = Age
        self.DeliveryPersonRating = DeliveryPersonRating
        self.VechileCondition = VechileCondition
        self.MultipleDeliveries = MultipleDeliveries
        self.Total_Ordering_PickupTime = Total_Ordering_PickupTime
        self.Year = Year
        self.Month = Month
        self.Day = Day 
        self.WeatherCondition= WeatherCondtion
        self.RoadTrafficCondition = RoadTrafficCondition
        self.Typeofvechile = Typeofvechile
        self.Typeoforder = Typeoforder
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age" : [self.Age],
                "Delivery_person_Ratings":[self.DeliveryPersonRating],
                "Weather_conditions" : [self.WeatherCondition],
                "Road_traffic_density" :[self.RoadTrafficCondition] ,
                "Vehicle_condition":[self.VechileCondition],
                "Type_of_order" : [self.Typeoforder],
                "Type_of_vehicle" : [self.Typeofvechile],
                "multiple_deliveries":[self.MultipleDeliveries],
                "Festival" :[self.Festival],
                "City" :[self.City],
                "Order_pickup_time":[self.Total_Ordering_PickupTime],
                "Year":[self.Year],
                "Month":[self.Month],
                "Day":[self.Day]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

        