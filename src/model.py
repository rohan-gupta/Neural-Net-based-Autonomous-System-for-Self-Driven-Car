import keras
from keras.layers import Dropout, Dense, Activation
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50, decode_predictions
import preparedata

#Prepare Model
def build():
    resnet = ResNet50(weights='imagenet')

    for layer in resnet.layers[1:45]:
        layer.trainable = False

    hidden = Dense(512, activation='relu')(resnet.output)
    hidden = Dropout(0.50)(hidden)
    hidden = Dense(256, activation='relu')(hidden)
    hidden = Dropout(0.50)(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    output = Dense(1)(hidden)

    model = Model(inputs=resnet.input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
