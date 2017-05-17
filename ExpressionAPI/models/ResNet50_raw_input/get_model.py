import tensorflow.contrib.keras as K
import tensorflow as tf

from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import layers
base_net = applications.resnet50.ResNet50( weights = 'imagenet' )
base_net.load_weights('./best_model.h5',by_name=True)
base_net.save_weights('./best_model_slim.h5')
