from flask import Flask ,request,render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predictdata",methods=['POST','GET'])
def predictdata():
    if request.method == 'GET':
        return render_template('index.html',results=0)

    else:
        ## Static Pass Values
        data = CustomData(
            Age=22,
            DeliveryPersonRating=3,
            VechileCondition=1,
            MultipleDeliveries= 1,
            Total_Ordering_PickupTime=20,
            Year=2023,
            Month=12,
            Day=22,
            WeatherCondtion='Sunny',
            RoadTrafficCondition='Low',
            Typeofvechile='electric_scooter',
            Typeoforder='Meal',
            Festival='No',
            City='Urban',
     
        )
         
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.Predict(final_new_data)
        return render_template('index.html',results = results[0])

if __name__ == "__main__":
    app.run(debug=True)