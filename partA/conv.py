

!pip install tensorflow-gpu

!nvidia-smi

!pip install wandb -qqq
import wandb
!wandb login fb3bb8a505ba908b667b747ed68e4b154b2f6fc5
from wandb.keras import WandbCallback
wandb.init(project="ConvolutionNN", entity="dl_assignment2")

from google.colab import drive
drive.mount('/content/gdrive')

import tensorflow as tf
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
          'nFilters' : {'values' : [32, 64, 128]},
          'filterSize' : {'values' : [32, 64]},
          'activationFuncs' : {'values' : ['sigmoid', 'tanh', 'relu']},
          'sizeMaxpool' : {'values' : [2, 3, 4, 5]},
          'nDense' :  {'values' : [16, 32, 64, 128]},
          'dataAugment' :  {'values' : [True]},
          'batchNormalization' : {'values' : [True]},
          'epochs' : {'values' : [10, 15]},
          'batchSize' : {'values' : [16, 32, 64, 128]},
          'learning_rate' : {'values' : [1e-2, 1e-3, 1e-4]},
          'dropout' :  {'values' : [0.2, 0.4, 0.5]},
          'optimizer': {'values' : ["Adam"]},
          'global_flattening_layer':{'values' : ['GlobalAveragePooling2D','GlobalMaxPool2D', 'Flatten']},
          'filterArrangement' : {'values' : ['equal','doubling','halving']},
          'convLayerSize' : {'values' : [5, 10, 15]},
          'denseLayerSize' : {'values' : [1]}
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
    
    
       


def buildModel(sizeFilters,nFilters,activationFuncs,sizeMaxpool,sizeDenseLayers, dropout, _globalLayer,filterArrangement,batchNormalization, dataAugment = True):
    
    #Model Characteristics
    #Note: batchNormalization
    convLayerSize = 5
    denseLayerSize = 1
    nClassifiers = 10

    if filterArrangement == 'doubling':
      for i in range(len(nFilters)-1):
        nFilters[i+1] = 2*nFilters[i]
    elif filterArrangement == 'halving':
        nFilters[i+1] = int(nFilters[i]/2)

   
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.Resizing(ImageSize[0], ImageSize[0]),
        ])
    
    if _globalLayer == "GlobalAveragePooling2D":
        globalLayer = keras.layers.GlobalAveragePooling2D()
    elif _globalLayer == "GlobalMaxPool2D":
        globalLayer = keras.layers.GlobalMaxPool2D()
    elif _globalLayer == "Flatten":
        globalLayer = keras.layers.Flatten()



    model = Sequential()
    inputLayer = keras.Input(shape=InputShape)
    model.add(inputLayer)
    if dataAugment:
      x = data_augmentation
      model.add(x)

      

    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))

    for i in range(convLayerSize):
      model.add(layers.Conv2D(nFilters[i], (sizeFilters[i], sizeFilters[i])))
      model.add(layers.Activation(activationFuncs[i]))
      model.add(layers.MaxPooling2D(pool_size=(sizeMaxpool[i],sizeMaxpool[i])))
        
    model.add(globalLayer)
    model.add(layers.Dropout(dropout))
    for i in range(denseLayerSize):
        model.add(layers.Dense(sizeDenseLayers[i], activation = "relu" ))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model

def getCallbacks(isWandBActive):
    callback = EarlyStopping(monitor='val_accuracy', patience=2)
    if isWandBActive:
        callbacks = [WandbCallback(),callback]
    else:
        callbacks = [callback]
    return callbacks


config_defaults = {
          'nFilters' : 32,
          'filterSize' : 5,
          'activationFuncs' : 'relu',
          'sizeMaxpool': 2,
          'nDense' : 64,
          'dataAugment' : True,
          'batchNormalization' : True,
          'epochs' : 10,
          'batchSize' : 64,
          'learning_rate' : 1e-3,
          'dropout' : 0.2,
          'seed' : 42,
          'optimizer':'Adam',
          'global_flattening_layer':'GlobalAveragePooling2D',
          'filterArrangement' : {'values' : ['equal','doubling','halving']},
          'convLayerSize' : 5,
          'denseLayerSize' : 1,
          }


isWandBActive = True

def run():
    convLayerSize = 5
    denseLayerSize = 1
    nClassifiers = 10

    if isWandBActive:
      wandb.init(config = config_defaults)
      config = wandb.config
      nFilters = [config.nFilters]*convLayerSize
      sizeFilters = [config.filterSize]*convLayerSize
      activationFuncs = [config.activationFuncs]*convLayerSize
      sizeMaxpool = [config.sizeMaxpool]*convLayerSize
      sizeDenseLayers = [config.nDense]*denseLayerSize
      dataAugment = config.dataAugment
      batchNormalization = config.batchNormalization
      epochs = config.epochs
      batchSize = config.batchSize
      learning_rate = config.learning_rate
      dropout = config.dropout
      seed = 42
      optimizer = config.optimizer
      global_flattening_layer = config.global_flattening_layer
      filterArrangement = config.filterArrangement

    else:
      nFilters = [32]*convLayerSize
      sizeFilters = [5]*convLayerSize
      activationFuncs = ["relu"]*convLayerSize
      sizeMaxpool = [2]*convLayerSize
      sizeDenseLayers = [64]*denseLayerSize
      dataAugment = True
      batchNormalization = True
      epochs = 10
      batchSize = 64
      learning_rate = 1e-3
      dropout = 0.2
      seed = 42
      optimizer = "Adam"
      global_flattening_layer = "GlobalAveragePooling2D"
      filterArrangement ="equal"

    callbacks = getCallbacks(isWandBActive)

    
    trainDataset, valDataset = loadData(TrainPath, batchSize=batchSize , typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=batchSize)
    valDataset = valDataset.prefetch(buffer_size=batchSize)
    model = buildModel(sizeFilters,nFilters,activationFuncs,sizeMaxpool,sizeDenseLayers, dropout, global_flattening_layer,filterArrangement,batchNormalization, dataAugment)
    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(trainDataset, epochs = config.epochs, validation_data=valDataset, callbacks=callbacks)
    

if isWandBActive:
    sweep_id = wandb.sweep(sweep_config, entity="dl_assignment2", project="ConvolutionNN")
    wandb.agent(sweep_id,function=run)
else:
    run()
