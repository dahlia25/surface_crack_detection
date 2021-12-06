# MobileNet Run

## File/Directory Structure

-Project
   -MobileNet ipynb file
   -data
      -Train   
         -Crack (Training Positive Data images)
         -Intact (Training Negative Data images)
      -Test
         -Crack
         -Intact	
         -"small_1000_set.pth" file for the augmented test data
		


## Required Python Packages

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model,layers
import tensorflow as tf
import keras
import torch

import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import os

import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import Callback,EarlyStopping
import tensorflow