

from google.colab import drive

!pip install wandb

import numpy as np

import wandb
from wandb.keras import WandbCallback
wandb.init(project="ConvolutionNN", entity="dl_assignment2")

from google.colab import drive
drive.mount('/content/gdrive')

import tensorflow as tf
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

sweep_config = {
      'method' : 'random',
      'metric' : {
          'name' : 'accuracy',
          'goal' : 'maximize'
      },
      'parameters' : {
          'nFilters' : {'values' : [128]},
          'convLayerSize' : {'values' : [5]},
          'learning_rate' : {'values' : [0.0001]},
          'maxPoolSize' : {'values' : [2]},
          'batchSize' : {'values' : [16]},
          'denseLayerSize' : {'values' : [3]},
          'filterSize' : {'values' : [64]},
          'epochs' : {'values' : [10]},
          'optimizer':{'values' : ['Adam']},
          'dropout ' : {'values' : [0.3]},
          'activationFuncs' : {'values' : ['tanh']},
          'global_flattening_layer':{'values' : ['Flatten']},
      }
      
}
sweep_id = wandb.sweep(sweep_config, entity="dl_assignment2", project="ConvolutionNN")

#Global Variables
TrainPath = "/content/gdrive/MyDrive/inaturalist_12K/train"
TestPath = "/content/gdrive/MyDrive/inaturalist_12K/val"
ImageSize = (256,256)
InputShape = (256,256,3)


'''
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels[0])  # int32
'''

def loadData(path, batchSize = 64, typeData = None):
    # Create Image Data Generator for Train Set
    image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    # Create Image Data Generator for Test/Validation Set
    test_data_gen = ImageDataGenerator(rescale = 1./255)

    
    if typeData == "train":
        trainDataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "training",  batch_size=batchSize, image_size = ImageSize)
        valDataset   = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "validation",  batch_size=batchSize, image_size = ImageSize)
        
        return trainDataset, valDataset

    elif typeData == "test":
        dataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337,batch_size=batchSize, image_size = ImageSize)
        return dataset
    
    
       


def buildModel(inputShape, dropout, _globalLayer, dataAugment = True,):
    
    #Model Characteristics
    convLayerSize = 5
    nFilters = [32]*convLayerSize
    filterSize = [5]*convLayerSize
    activationFuncs = ["relu"]*convLayerSize
    maxPoolSize = [2]*convLayerSize
    denseLayerSize = 1
    nDense = [64]*denseLayerSize
    nClassifiers = 10

    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        ])
    
    if _globalLayer == "GlobalAveragePooling2D":
        globalLayer = keras.layers.GlobalAveragePooling2D()
    elif _globalLayer == "GlobalMaxPool2D":
        globalLayer = keras.layers.GlobalMaxPool2D()
    elif _globalLayer == "Flatten":
        globalLayer = keras.layers.Flatten()



    model = Sequential()
    inputLayer = keras.Input(shape=InputShape)
    if dataAugment:
      x = data_augmentation
      model.add(x)
    else:
      model.add(inputLayer)

    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    '''for i in range(convLayerSize):
        model.add(layers.Conv2D(nFilters[i], filterSize[i], strides=1, activation=activationFuncs[i]))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(maxPoolSize[i]))'''
        
    model.add(globalLayer)

    for i in range(denseLayerSize):
        model.add(layers.Dense(nDense[i], activation = "relu" ))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model




def makePredictions(model):
    batchSize = 64
    testDataset = loadData(TestPath,batchSize=batchSize, typeData= "test")
    testDataset = testDataset.prefetch(buffer_size=batchSize)
    loss, acc = model.evaluate(testDataset)
    predictions = model.predict(testDataset)

    return loss, acc, predictions


def getCallbacks(isWandBActive):
    callback = EarlyStopping(monitor='val_accuracy', patience=2)
    if isWandBActive:
        callbacks = [WandbCallback(),callback]
    else:
        callbacks = [callback]
    return callbacks


config_defaults = {
          'epochs' : 10,
          'batchSize' : 16,
          'convLayerSize' : 5,
          'learning_rate' : 0.0001,
          'activationFuncs' : 'tanh',
          'dropout' : 0.3,
          'seed' : 42,
          'nFilters' : 128,
          'filterSize' : 64,
          'optimizer':'Adam',
          'global_flattening_layer':'Flatten',
          'denseLayerSize' : 3,
          'maxPoolSize' : 2,
          'nDense' : 64
        }


isWandBActive = True

def run():
    if isWandBActive:
      wandb.init(config = config_defaults)
      config = wandb.config
      epochs = config.epochs
      batchSize = config.batchSize
      convLayerSize = config.convLayerSize
      learning_rate = config.learning_rate
      activationFuncs = config.activationFuncs
      dropout = config.dropout
      seed = config.seed
      nFilters = config.nFilters
      filterSize = config.filterSize
      optimizer = config.optimizer
      global_flattening_layer = config.global_flattening_layer
      denseLayerSize = config.denseLayerSize
      maxPoolSize = config.maxPoolSize
      nDense = config.nDense

    else:
      epochs = 10,
      batchSize = 16,
      convLayerSize = 5,
      learning_rate = 0.0001,
      activationFuncs = "tanh",
      dropout = 0.3,
      seed = 42,
      nFilters = 128,
      filterSize = 64,
      optimizer = "Adam",
      global_flattening_layer = "Flatten",
      denseLayerSize = 3,
      maxPoolSize = 2,
      nDense = 64

    callbacks = getCallbacks(isWandBActive)
    

    trainDataset, valDataset = loadData(TrainPath, batchSize=config.batchSize , typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=config.batchSize)
    valDataset = valDataset.prefetch(buffer_size=config.batchSize)
    model = buildModel(InputShape, dropout=config.dropout, _globalLayer=config.global_flattening_layer, dataAugment=True)
    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
  
    return model
model = run()

#wandb.agent(sweep_id,function=run)

"""Q4 Implementation

"""

#batchSize = 64
testDataset = loadData(TestPath,batchSize=batchSize, typeData= "test")
testDataset = testDataset.prefetch(buffer_size=batchSize)
loss, acc = model.evaluate(testDataset)
predictions = model.predict(testDataset)

np.argmax(predictions, axis= 1).shape

classNames = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

for images, labels in testDataset.take(1):
  print(labels.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for images, labels in testDataset.take(1):
    for i in range(30):
        ax = plt.subplot(10, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classNames[np.argmax((labels[i]))])
        plt.axis("off")

model.summary()

keras.layers.Conv2D.get_output_at

for layer in model.layers:
  if 'conv' in layer.name:
    #filters, biases = layer.get_weights()
    #print(layer.name, filters.shape)

    model = keras.Model(inputs=model.inputs, outputs=layer.output)
    feature_maps = model.predict(testDataset.take(1))
    break

feature_maps.shape



sizeFilter = feature_maps.shape[-1]
square = 8
ix = 1
for _ in range(square):
	for _ in range(4):
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()
