import numpy as np
from flask import Flask, request, jsonify
import pickle
from flask_cors import cross_origin
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    
    data = request.get_json(force=True)
    sales_lag_1 = data['sales_lag_1']
    sales_lag_2 = data['sales_lag_2']
    input_features = np.array([[sales_lag_1, sales_lag_2]])
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
