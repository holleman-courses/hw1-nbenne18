import tensorflow as tf
import keras
from keras import Input, layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

import hw1_complete as hw



model1 = hw.build_model1()
model2 = hw.build_model2()
model3 = hw.build_model3()



# Load CIFAR 10 dataset
(train_images, train_labels), (test_images, test_labels) = \
  tf.keras.datasets.cifar10.load_data()

train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()

train_images = train_images / 255.0
test_images  = test_images  / 255.0

  
try:
  model50k = tf.keras.models.load_model("best_model.h5")
except:
  print("Failure loading best_model.h5")
else:
  print("Model50k loaded successfully.")



model1_layers = ['Flatten', 'Dense', 'Dense', 'Dense', 'Dense']
model1_params = [0, 393344, 16512, 16512, 1290]

model2_layers = ['Conv2D', 'BatchNormalization', 'Conv2D', 'BatchNormalization', 'Conv2D',
                 'BatchNormalization', 'Conv2D', 'BatchNormalization', 'Conv2D',
                 'BatchNormalization', 'Conv2D', 'BatchNormalization', 'Flatten', 'Dense'
                 ]
model2_params = [896, 128, 18496, 256, 73856, 512, 147584, 512, 147584, 512, 147584, 512, 0, 81930]

model3_layers = ['SeparableConv2D', 'BatchNormalization', 'SeparableConv2D', 'BatchNormalization',
                 'SeparableConv2D', 'BatchNormalization', 'SeparableConv2D', 'BatchNormalization',
                 'SeparableConv2D', 'BatchNormalization', 'SeparableConv2D', 'BatchNormalization',
                 'Flatten', 'Dense'
                 ]
model3_params = [155, 128, 2400, 256, 8896, 512, 17664, 512, 17664, 512, 17664, 512, 0, 81930]


def count_layers(model, layer_type):
  lyr_count = 0
  for l in model.layers: 
    if l.__class__.__name__ == layer_type:
      lyr_count +=1 
  return lyr_count

def get_model_info(model):
  # Create the list of layer info
  layer_types = []
  param_counts = []
  for layer in model.layers:
    layer_types.append(layer.__class__.__name__)
    param_counts.append(layer.count_params())
    
  return layer_types, param_counts
  
def test_freebie():
  assert 1 == 1

def test_model1_layers():
  layer_names, _ = get_model_info(model1)
  assert layer_names == model1_layers

def test_model1_params():
  _, param_counts = get_model_info(model1)
  assert param_counts == model1_params

def test_model2_layers():
  layer_names, _ = get_model_info(model2)
  assert layer_names == model2_layers

def test_model2_params():
  _, param_counts = get_model_info(model2)
  assert param_counts == model2_params

def test_model3_layers():
  layer_names, _ = get_model_info(model3)
  assert layer_names == model3_layers

def test_model3_params():
  _, param_counts = get_model_info(model3)
  assert param_counts == model3_params
  
   
def test_model50k_params():
  assert model50k.count_params() <= 50000

def test_model50k_acc50():
  loss, acc = model50k.evaluate(test_images, test_labels)
  assert acc >= 0.50

def test_model50k_acc60():
  loss, acc = model50k.evaluate(test_images, test_labels)
  assert acc >= 0.60


  
