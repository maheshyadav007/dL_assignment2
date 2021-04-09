
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

sweep_config = {
      'method' : 'random',
      'metric' : {
          'name' : 'accuracy',
          'goal' : 'maximize'
      },
      'parameters' : {
          'nFilters' : {'values' : [32, 64, 128]},
          'convLayerSize' : {'values' : [5, 10, 15]},
          'learning_rate' : {'values' : [1e-2, 1e-3, 1e-4]},
          'maxPoolSize' : {'values' : [2, 3, 4, 5]},
          'batchSize' : {'values' : [16, 32, 64, 128]},
          'denseLayerSize' : {'values' : [1, 2, 3, 4, 5]},
          'filterSize' : {'values' : [32, 64]},
          'epochs' : {'values' : [5, 10, 15]},
          'optimizer':{'values' : ['Adam']},
          'dropout ' : {'values' : [0.3, 0.4, 0.5]},
          'activationFuncs' : {'values' : ['sigmoid', 'tanh', 'relu']},
          'global_flattening_layer':{'values' : ['GlobalAveragePooling2D','GlobalMaxPool2D', 'Flatten']},
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
    #print(inputLayer.shape)
    if dataAugment:
      x = data_augmentation(inputLayer)
      #print(x.shape)
      model.add(x)
    else:
      model.add(inputLayer)

    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))

    for i in range(convLayerSize):
        model.add(layers.Conv2D(nFilters[i], filterSize[i], strides=1, activation=activationFuncs[i]))
        model.add(layers.MaxPooling2D(maxPoolSize[i]))
        
    model.add(globalLayer)

    for i in range(denseLayerSize):
        model.add(layers.Dense(nDense[i], activation = "relu" ))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model



def run():

    config_defaults = {
      'epochs' : 10,
      'batchSize' : 64,
      'convLayerSize' : 5,
      'learning_rate' : 1e-3,
      'activationFuncs' : 'relu',
      'dropout' : 0.2,
      'seed' : 42,
      'nFilters' : 32,
      'filterSize' : 5,
      'optimizer':'Adam',
      'global_flattening_layer':'GlobalAveragePooling2D',
      'denseLayerSize' : 1,
      'nDense' : 64
      }
    wandb.init(config=config_defaults)
    config = wandb.config
    

    trainDataset, valDataset = loadData(TrainPath, batchSize=config.batchSize , typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=config.batchSize)
    valDataset = valDataset.prefetch(buffer_size=config.batchSize)
    model = buildModel(InputShape, dropout=config.dropout, _globalLayer=config.global_flattening_layer, dataAugment=True)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    model.fit(trainDataset, epochs = config.epochs, validation_data=valDataset, callbacks=[WandbCallback(early_stopper)])
    wandb.log({ 'accuracy' : accuracy})
    wandb.log({ 'loss' : loss})

run()

wandb.agent(sweep_id,function=run)
