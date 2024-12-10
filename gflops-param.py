from __future__ import print_function
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.preprocessing.image import img_to_array
#import imutils
import cv2
from keras.models import load_model
import numpy as np

from utils2 import get_initial_weights
#from layers2 import BilinearInterpolation
from keras_layer_normalization import LayerNormalization
import tensorflow as tf
from keras_flops import get_flops

det_time = 0
det_time_tot = 0
class_time = 0
class_time_tot = 0
n = 0

emotion_model_path = 'models/Model-1.h5'

emotion_classifier = load_model(emotion_model_path, compile=False, custom_objects={'LayerNormalization':LayerNormalization, 'tf':tf})
emotion_classifier.summary()
print("Loaded model from disk")

flops = get_flops(emotion_classifier, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.04} G")

