import preparedata
import keras
from keras.models import load_model

def train(model):
    #Prepare Training Data
    data = preparedata.preparedata(path='../data/train/')
    imagedata = data[0]
    steeringangle = data[1]

    x_train, y_train = imagedata, steeringangle
    model.fit(x_train, y_train, validation_split=0.20, batch_size=50, epochs=10)
    model.save('../model.h5')
