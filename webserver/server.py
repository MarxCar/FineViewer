# Ensure CPU Only 
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import StyleGAN
sys.path.append("..")
from StyleGAN import WGAN


# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import base64
import numpy as np

# instantiate flask 
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = WGAN(lr = 0.0003, silent=False)
model.load(61)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        print("REEE")
        x = np.genfromtxt(params["image"])
        with graph.as_default():
        
            data["prediction"] = base64.b64encode(model.imageFromLatent(x))
            data["success"]
            
        #x=pd.DataFrame.from_dict(params, orient='index').transpose()
        #with graph.as_default():
        #    data["prediction"] = str(model.predict(x)[0][0])
        #    data["success"] = True
    else:
        print("BEEEE")

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')