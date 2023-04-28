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
        req = request.form
        print(req)
        ## Static Pass Values
        data = CustomData(
            Age=req['Age'],
            DeliveryPersonRating=req['DeliveryPersonRating'],
            VechileCondition=req['VechileCondition'],
            MultipleDeliveries= req['MultipleDeliveries'],
            Total_Ordering_PickupTime=req['Total_Ordering_PickupTime'],
            Year=req['Year'],
            Month=req['Month'],
            Day=req['Day'],
            WeatherCondtion=req['Weather'],
            RoadTrafficCondition=req['RoadTrafficCondition'],
            Typeofvechile=req['Typeofvechile'],
            Typeoforder=req['Typeoforder'],
            Festival=req['Festival'],
            City=req['City'],
     
        )
         
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.Predict(final_new_data)
        return render_template('index.html',results = results[0])

if __name__ == "__main__":
    app.run(debug=True)