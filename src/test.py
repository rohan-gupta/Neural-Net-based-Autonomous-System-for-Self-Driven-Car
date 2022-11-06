import preparedata
import pandas as pd
import numpy as np
import keras
from keras.models import load_model

def test():
    data = preparedata.preparedata(path='../data/test/')

    file = data[0]
    imagedata = data[1]
    actualangle = data[2].flatten().tolist()

    model = load_model('../data/model.h5')
    result = model.predict(imagedata)

    predictedangle = []

    for i in range(len(file)):
        predictedangle.append(result[i][0])

    df = {'File':file, 'Predicted Angles':predictedangle, 'Actual Angles':actualangle}
    df = pd.DataFrame(data=df)
    df = df[['File','Predicted Angles','Actual Angles']]
    df.to_csv('../data/result.csv')
