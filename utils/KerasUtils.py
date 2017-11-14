from keras.models import Model
from keras.models import load_model
import numpy as np

def evaluateModel(event, model):
    x=np.reshape(event, (1, event.shape[0], event.shape[1]))
    prediction=model.predict(x,verbose=0)
    return prediction

def loadModel(weightFile, customLayers):


    model = load_model(weightFile, custom_objects=customLayers)
    return model

def getIntermediateLayer(model, layerName):

    intermediateLayer = Model(inputs=model.input,
                              outputs=model.get_layer(layerName).output)
    return intermediateLayer

def getLayerCollection(model, layerNames):
    layers={}
    for layer in layerNames:
        layers[layer]=getIntermediateLayer(model,layer)

    return layer
