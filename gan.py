import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
#from tensorflow.keras import layers
import time

from IPython import display

# import numpy
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

