import glob as glob
import numpy as np
import pandas as pd
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

#Data
def preparedata(path):
    df = pd.read_csv(path+'interpolated.csv')
    df = df[df['frame_id']=='center_camera'][['filename','angle']]

    #Shuffling the data
    # df = df.sample(frac=1)

    filename = list(df['filename'])
    steeringangle = np.array(list(df['angle']))
    steeringangle = np.reshape(steeringangle, (len(steeringangle),1))

    imagedata = []

    for imagepath in filename:

        img = image.load_img(path+imagepath, target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        imagedata.append(img[0])

    imagedata = np.array(imagedata)

    return [filename, imagedata, steeringangle]
