import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the model
model=pickle.load(open('regressor.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)

    # Convert input to numpy array and reshape
    input_data = np.array(list(data.values())).reshape(1, -1)
    print(input_data)

    # Transform using scaler
    new_data = scaler.transform(input_data)

    # Predict using model
    output = model.predict(new_data)
    print(output[0])

    # Return JSON response
    return jsonify({"Prediction": float(output[0])})

if __name__=="__main__":
    app.run(debug=True)


