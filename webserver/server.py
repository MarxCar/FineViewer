# Ensure CPU Only 
import os
import sys
import io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
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
from PIL import Image

# instantiate flask 
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = WGAN(lr=0.0003, silent=False)
model.load(61)


# define a predict function as an endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = flask.request.files

    # if parameters are found, return a prediction
    print(flask.request.files)
    if (params != None):

        x = np.load(params["latent"])
        bytesIO = io.BytesIO()

        with graph.as_default():
            image = Image.fromarray(model.imageFromLatent(x).reshape(64, 64, 3))
            image.convert("RGB").save(bytesIO, format="PNG")
            bytesIO.seek(0, 0)
            return flask.send_file(bytesIO, as_attachment=False, mimetype="image/png")
    else:
        return flask.jsonify(data)


@app.route("/interpolation", methods=["POST"])
def interpolation():
    params = flask.request.files
    if params != None:
        x1 = np.load(params["latent1"])
        x2 = np.load(params["latent2"])
        n = int(flask.request.values["n"])
        if n > 15:
            return TypeError("N cannot be greater than 15")
        bytesIO = io.BytesIO()

        with graph.as_default():
            image = Image.fromarray(model.interpolation(x1, x2, n))
            print(image)
            image.convert("RGB").save(bytesIO, format="PNG")
            bytesIO.seek(0, 0)
            return flask.send_file(bytesIO, as_attachment=False, mimetype="image/png")
    else:
        return TypeError("Didn't send right parameters")


# start the flask app, allow remote connections 
app.run(host='0.0.0.0')
