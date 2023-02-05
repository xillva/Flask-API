import pickle as pk
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd
import os

path = os.path.realpath(os.path.dirname(__file__))
path = path.replace("\\", "/")

with open(path + "/rf.pkl", "rb") as model:
    model = pk.load(model)
    
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """
    Example endpoint returning a prediction of iris
    ---
    parameters:
        - name: sl
          in: query
          type: number
          required: true
        - name: sw
          in: query
          type: number
          required: true
        - name: pl
          in: query
          type: number
          required: true
        - name: pw
          in: query
          type: number
          required: true
    responses:
        200:
            description: "0: Iris-setosa, 1: Iris-versicolor, 2: Iris-virginica"
    """
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')
    
    prediction = model.predict(np.array([[sl, sw, pl, pw]]))
    return str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_iris_file():
    """
    Example endpoint returning a prediction of iris
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: "0: Iris-setosa, 1: Iris-versicolor, 2: Iris-virginica"
    """
    input_data = pd.read_csv(request.files.get('input_file'), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)