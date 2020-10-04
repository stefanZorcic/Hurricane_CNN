# Hurricane_CNN
Implementing a conventional neural networks in python to locate hurricanes

conventional-neural-network.py uses artifical intellegence and deep learning to determine location of hurricanes on the world map provided my NASA.

Line 38 of the code has a variable "date" that is responsible for the date that the CNN will anaylses to find hurricanes, this date is editable but follow the format provided in the variable.

This program run on python3.6, it is not backwards compatible with python2 or any version of python2.

Before running the code it is important for the modules of the code to be installed on your system and for the code to be able to acess these modules.

Such modules that the code needs will be listed below

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

View offical documentation pertaining to each of the modules to see the best suited way to download the modules.

The code has been tested and will run if the condition has been meet. In the event of an error please email codepitt@gmail.com, the email will resolve any error that this code my have produced.

This program is very resource intensive, therefore the minumum system requirments for such the program to be run smoothly is a 8GB RAM system and 2.5GHZ dual core processor.

The recommended system requirments to run this program is 16GB DDR4 RAM and 2GHZ quad core and a dedicated graphics card.

IF YOU DO NOT HAVE THE MINIMUM SYSTEM REQUIRMENTS PLEASE DO NOT ATTEMPT RUNNING THE CODE. IT WILL CRASH YOUR SYSTEM.

Python Module "Tensorflow" is optimized with GPU's, hence to improve preformace consider switching the GPU of your system.

Upon running the code a GUI will appear, be paitent it will appear. When the GUI appears 3 inputs will be on the right of the map, the recommended values are below. The only datatype that should be entered into the inputs are integers >1 and <1000.

Recommended GUI Input Integers:
Data Points: 1000, DO NOT INCREASE THIS THE VALUE IN THIS FEILD IT WILL CRASH YOUR SYSTEM
Resolution: 200, This can be increased to 1000, however the higher this number the more resource intensive the code will be
Epochs: 100, This is important to converge on model accuracy, low number of epochs will cause the system to be able to identify hurricanes

Upon pressing train the AI will train itself with the given values and find hurricanes on the date provided

This model was able to acheive 70% accuracy on low settings, I was unable to increase accuracy for fear of crashing my system.

Enjoy! :)


