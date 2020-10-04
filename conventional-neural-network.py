import cv2 as cv2
from matplotlib import pyplot as plt
import os
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from io import BytesIO
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import requests
import cv2 as cv
from io import BytesIO
from matplotlib.pyplot import imshow
from PIL import Image
import requests
import io
import imgkit




print("Imported")

global model, img_data, hurricanes, date



date = '2019-09-01T00:00:00Z'



def model_train(res1,num1,epoc):

    data=[0]*0
    key=[0]*0
    res= (res1,res1)
    num=num1
    re = res1
    ep = epoc

    for i in range(num):

      x = str(os.listdir("yes")[i])
      x = "yes/" + x
      img = cv2.imread(x, 1)
      img=img.copy()

      maxsize = res
      img = cv2.resize(img,maxsize)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      dst = cv2.fastNlMeansDenoisingColored(img,None, 3, 3, 1)

      #plt.imshow(dst)
      #plt.show()

      thresh = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

      #plt.imshow(thresh)
      #print(i, time.strftime("%M:%S"))
      data.append(thresh)
      key.append(1)
      #plt.show()

    for i in range(num):

      x = str(os.listdir("no")[i])
      x = "no/" + x
      img = cv2.imread(x, 1)
      img=img.copy()

      maxsize = res
      img = cv2.resize(img,maxsize)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      dst = cv2.fastNlMeansDenoisingColored(img,None, 3, 3, 1)

      #plt.imshow(dst)
      #plt.show()

      thresh = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

      #plt.imshow(thresh)
      #print(i, time.strftime("%M:%S"))
      data.append(thresh)
      key.append(0)
      #plt.show()

    #print(len(data))

    #print(data[1])

    data, key = shuffle(data, key)

    #print(data, key)

    #print(thresh)

    s = int(len(data)*0.1)
    if s==0:
      s+=1

    #print(s)

    X_train = data[s:]
    Y_train = key[s:]
    X_test = data[:s]
    Y_test = key[:s]

    print("X_train: " + str(len(X_train)))
    print("X_Test: " + str(len(X_test)))

    Y_train=np.array(Y_train)
    X_train=np.array(X_train)

    r = np.shape(X_train)[0]*np.shape(X_train)[1]*np.shape(X_train)[2]*np.shape(X_train)[3]
    r = r/np.shape(X_train)[0]
    r = int(r)

    #X_train = X_train.reshape(np.shape(X_train)[0],r)
    X_test = np.array(X_test)
    #X_test = X_test.reshape(np.shape(X_test)[0],r)


    print("Modelling")




    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=(np.shape(X_train)[1:])))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layer
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    # print the summary of the model architecture
    model.summary()
    # training the model using adam optimizer
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


    Y_train = np.asarray(Y_train).astype('float32').reshape((int(len(X_train)),1))
    Y_test = np.asarray(Y_test).astype('float32').reshape((int(len(Y_train)),1))

    print("Fitting...")
    #model.train(X_train,Y_train)
    model.fit(X_train, Y_train,validation_data=(X_test, Y_test),epochs=ep)
    #model.save("results/cifar10-model-v1.h5")



url = "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&version=1.3.0&crs=EPSG:4326&transparent=false&WIDTH={}&HEIGHT={}&BBOX={}&FORMAT=image/tiff&TIME={}"
KM_PER_DEG_AT_EQ = 111.

def calculate_width_height(extent, resolution):
    """
    extent: [lower_latitude, left_longitude, higher_latitude, right_longitude], EG: [51.46162974683544,-22.94768591772153,53.03698575949367,-20.952234968354432]
    resolution: represents the pixel resolution, i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 5, 10]
    """
    lats = extent[::2]
    lons = extent[1::2]
    km_per_deg_at_lat = KM_PER_DEG_AT_EQ * np.cos(np.pi * np.mean(lats) / 180.)
    width = int((lons[1] - lons[0]) * km_per_deg_at_lat / resolution)
    height = int((lats[1] - lats[0]) * KM_PER_DEG_AT_EQ / resolution)
    print(width, height)
    return (width, height)

def modis_url(time, extent, resolution):
    """
    time: utc time in iso format EG: 2020-02-19T00:00:00Z
    extent: [lower_latitude, left_longitude, higher_latitude, right_longitude], EG: [51.46162974683544,-22.94768591772153,53.03698575949367,-20.952234968354432]
    resolution: represents the pixel resolution, i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 5, 10]
    """
    width, height = calculate_width_height(extent, resolution)
    extent = ','.join(map(lambda x: str(x), extent))
    return (width, height, url.format(width, height, extent, time))


def world_map():
    width, height, url = modis_url(date, [-90,-180,90,180], 50)
    response = requests.get(url)
    img = BytesIO(response.content)

    return img



def map_section(res1):
    img_data = [0] * 0
    def img_preprocessing(image_stream):
        maxsize = (res1, res1)
        # img1 = img1.seek(0)
        # image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.COLOR_BGR2RGB)

        img = cv2.resize(img, maxsize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 1)

        thresh = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # plt.imshow(thresh)
        # plt.show()

        img_data.append(thresh)


    def pic(x1, y1):
        URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&version=1.3.0&crs=EPSG:4326&transparent=false&WIDTH={}&HEIGHT={}&BBOX={}&FORMAT=image/tiff&TIME={}"
        KM_PER_DEG_AT_EQ = 111.

        def calculate_width_height(extent, resolution):
            """
            extent: [lower_latitude, left_longitude, higher_latitude, right_longitude], EG: [51.46162974683544,-22.94768591772153,53.03698575949367,-20.952234968354432]
            resolution: represents the pixel resolution, i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 5, 10]
            """
            lats = extent[::2]
            lons = extent[1::2]
            km_per_deg_at_lat = KM_PER_DEG_AT_EQ * np.cos(np.pi * np.mean(lats) / 180.)
            width = int((lons[1] - lons[0]) * km_per_deg_at_lat / resolution)
            height = int((lats[1] - lats[0]) * KM_PER_DEG_AT_EQ / resolution)
            print(width, height)
            return (width, height)

        def modis_url(time, extent, resolution):
            """
            time: utc time in iso format EG: 2020-02-19T00:00:00Z
            extent: [lower_latitude, left_longitude, higher_latitude, right_longitude], EG: [51.46162974683544,-22.94768591772153,53.03698575949367,-20.952234968354432]
            resolution: represents the pixel resolution, i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 5, 10]
            """
            width, height = calculate_width_height(extent, resolution)
            extent = ','.join(map(lambda x: str(x), extent))
            return (width, height, URL.format(width, height, extent, time))

        x = 90 - (x1 * 10)
        y = 180 - (y1 * 10)

        width, height, url = modis_url(date, [x - 10, y - 10, x, y], 5)
        response = requests.get(url)

        img = BytesIO(response.content)

        Image.open(img)
        return img


    for y in range(36):
        for x in range(18):
            img_preprocessing(pic(x, y))

    img_data = np.array(img_data)

    hurricanes=[0]*0

    for i in range(len(img_data)):
        r = img_data[i]
        r = r.reshape(200, 200, 3)
        #plt.imshow(r)
        #plt.show()
        r = np.expand_dims(r, axis=0)
        #print(model.predict(r))
        #print(sum(model.predict(r)[0]))

        if int(sum(model.predict(r)[0]))>=1:
            hurricanes.append(i)

import tkinter as tk
from PIL import ImageTk
#import tkinter import

window = tk.Tk()

greeting = tk.Label(text="Hello, Tkinter")

img = ImageTk.PhotoImage(Image.open(world_map()))

panel = tk.Label(window, image = img)
a = tk.Label(window, text="AI Training")

def v():
    #print(int(res.get()),int(epo.get()),int(datap.get()))
    model_train(int(res.get()),int(datap.get()),int(epo.get()))
    map_section(int(res.get()))

    if len(hurricanes)>0:
        for i in range(len(hurricanes)):
            Yh = int(hurricanes[i]/36)+1
            Xh = hurricanes[i]-int(hurricanes[i]/36)
            s = str("Potential Hurricane at: " + str(Yh*10) + " Longitude and " + str(Xh*10) + " Latitude.")
            b = tk.Label(window,s)
            b.place(x=800,y=90+20*(i+1))


n1 = tk.IntVar()
n2 = tk.IntVar()
n3 = tk.IntVar()

window.title(str("Date: " + date))

res = tk.Entry (window,textvariable=n1)
res.place(x=900,y=20)
b = tk.Label(window, text="Resolution:")
b.place(x=800,y=30)
epo = tk.Entry (window,textvariable=n2)
epo.place(x=900,y=40)
c = tk.Label(window, text="Epochs:")
c.place(x=800,y=50)
datap = tk.Entry (window,textvariable=n3)
datap.place(x=900,y=60)
d = tk.Label(window, text="Data Points:")
d.place(x=800,y=70)
bu = tk.Button(window, text="Train",command=v)
bu.place(x=800,y=90)


panel.place(x=0,y=0)
a.place(x=800,y=0)

window.mainloop()
