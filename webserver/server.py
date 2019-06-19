# Ensure CPU Only
import os
import sys
import io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import StyleGAN
sys.path.append("..")
from StyleGAN import WGAN, noise

# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import base64
import numpy as np
from PIL import Image
from flask_cors import CORS, cross_origin

# instantiate flask
app = flask.Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = WGAN(lr=0.0003, silent=False)
model.load(61)


# define a predict function as an endpoint
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    data = {"success": False}

    params = flask.request.values


    if (params != None):

        x = np.frombuffer(base64.decodebytes(params["latent"]))
        if x.shape == 2:
            x = x[0]
        bytesIO = io.BytesIO()

        with graph.as_default():
            image = Image.fromarray(model.imageFromLatent(x).reshape(64, 64, 3))
            image.convert("RGB").save(bytesIO, format="PNG")
            bytesIO.seek(0, 0)
            data["image"] = base64.b64encode(bytesIO.getvalue())
            return flask.jsonify(data)
    else:
        return flask.jsonify(data)


@app.route("/interpolation", methods=["POST"])
@cross_origin()
def interpolation():
    data = {"success": False}
    params = flask.request.values
    if params != None:
        x1 = np.frombuffer(base64.decodebytes(params["latent"]))
        x2 = np.frombuffer(base64.decodebytes(params["latent"]))
        n = int(params["n"])
        if n > 15:
            return TypeError("N cannot be greater than 15")
        bytesIO = io.BytesIO()

        with graph.as_default():
            image = Image.fromarray(model.interpolation(x1, x2, n))

            image.convert("RGB").save(bytesIO, format="PNG")
            bytesIO.seek(0, 0)
            data["image"] = base64.b64encode(bytesIO.getvalue())
            return flask.jsonify(data)
    else:
        return flask.jsonify(data)

@app.route("/randomFace", methods=["GET"])
@cross_origin()
def randomFace():
    bytesIO = io.BytesIO()
    data = {"success": True}

    with graph.as_default():
        image = Image.fromarray(model.imageFromLatent(noise(1)))
        image.convert("RGB").save(bytesIO, format="PNG")
        data["image"] = base64.b64encode(bytesIO.getvalue());
        return flask.jsonify(data)

@app.route("/randomLatent", methods=["GET"])
@cross_origin()
def randomLatent():
    data = {"success": True}
    data["latent"] = base64.b64encode(noise(1))
    return flask.jsonify(data)


@app.route("/changeLatent", methods=["POST"])
@cross_origin()
def changeLatent():
    data = {"success": False}
    if flask.request.files != None:
        latent = np.frombuffer(base64.decodebytes(flask.values["latent"]))[0]
        dim = int(flask.request.values["dimension"])
        operator = flask.request.values["operator"]
        bytesIO = io.BytesIO()
        with graph.as_default():
            if operator == "add":
                print(latent.shape)
                image = model.addToLatent(latent, dim)
            elif operator == "sub":
                image = model.subtractFromLatent(latent, dim)
            else:
                return flask.jsonify(data)
            image = Image.fromarray(image)
            image.convert("RGB").save(bytesIO, format="PNG")
            bytesIO.seek(0, 0)
            data["latent"] = base64.b64encode(bytesIO.getvalue())
            return flask.jsonify(data)

# start the flask app, allow remote connections
app.run(host='0.0.0.0')
