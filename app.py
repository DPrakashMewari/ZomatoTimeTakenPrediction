from flask import Flask ,request,render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder

app = Flask(__name__)

@app.route("/")
def home():
    return "Home"

@app.route("/predictdata",methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        pass
    


if __name__ == "__main__":
    app.run()