
!pip install tensorflow-gpu

!nvidia-smi

!pip install wandb -qqq
import wandb
!wandb login fb3bb8a505ba908b667b747ed68e4b154b2f6fc5
wandb.init(project="ConvolutionalNN", entity="cs20m040")

from google.colab import drive
drive.mount('/content/gdrive')


sweep_config = {
      'method' : 'random',
      'metric' : {
          'name' : 'accuracy',
          'goal' : 'maximize'
      },
      'parameters' : {
          'nFilters' : {'values' : [32, 64, 128]},
          'convLayerSize' : {'values' : [5, 10, 15]},
          'learning_rate' : {'values' : [1e-2, 1e-3]},
          'maxPoolSize' : {'values' : [2, 3, 4, 5]},
          'batchSize' : {'values' : [16, 32, 64, 128]},
          'denseLayerSize' : {'values' : [1, 2, 3, 4, 5]},
          'filterSize' : {'values' : [32, 64]},
          'epochs' : {'values' : [5, 10, 15]},
          'Dropout' : {'values' : [0.3, 0.4, 0.5]},
          'activationFuncs' : {'values' : ['sigmoid', 'tanh', 'relu']}
      }
      
}
sweep_id = wandb.sweep(sweep_config, entity="cs20m040", project="ConvolutionalNN")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Global Variables
TrainPath = "/content/gdrive/MyDrive/Colab Notebooks/inaturalist_12K/train"
TestPath = "/content/gdrive/MyDrive/Colab Notebooks/inaturalist_12K/val"
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
        
        # Create Image Data Augmentation for Train Set
        #trainDataset = image_gen.flow_from_directory(TrainPath, target_size=ImageSize, color_mode='grayscale', class_mode='categorical', batch_size=batchSize)

        # Create Image Data Augmentation for Validation Set
        #valDataset = test_data_gen.flow_from_directory(TrainPath, target_size=ImageSize, color_mode='grayscale', class_mode='categorical', batch_size=batchSize)

        return trainDataset, valDataset

    elif typeData == "test":
        dataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337,batch_size=batchSize, image_size = ImageSize)
        # Create Image Data Augmentation for Test Set
        #dataset = test_data_gen.flow_from_directory(TestPath, target_size=ImageSize, color_mode='grayscale', shuffle=False, class_mode='categorical', batch_size=batchSize)
        return dataset
    
    
       


def buildModel(inputShape):
    
    #Model Characteristics
    convLayerSize = 5
    nFilters = [32]*convLayerSize
    filterSize = [5]*convLayerSize
    activationFuncs = ["relu"]*convLayerSize
    maxPoolSize = [2]*convLayerSize
    denseLayerSize = 1
    nDense = [64]*denseLayerSize
    nClassifiers = 10

    model = Sequential()
    model.add(keras.Input(shape=inputShape))
    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))

    for i in range(convLayerSize):
        model.add(layers.Conv2D(nFilters[i], filterSize[i], strides=1, activation=activationFuncs[i]))
        model.add(layers.MaxPooling2D(maxPoolSize[i]))
        
    model.add(Flatten())

    for i in range(denseLayerSize):
        model.add(layers.Dense(nDense[i], activation = "relu" ))
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model


#model.add(layers.GlobalMaxPooling2D())
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_valid, y_valid),callbacks=[checkpointer])

def run():

    config_defaults = {
      'epochs' : 10,
      'batchSize' : 64,
      'convLayerSize' : 5,
      'learning_rate' : 1e-3,
      'activationFuncs' : 'relu',
      'Dropout' : 0.5,
      'seed' : 42,
      'nFilters' : 32,
      'filterSize' : 5,
      'denseLayerSize' : 1,
      'nDense' : 64
      }
    wandb.init(config=config_defaults)
    config = wandb.config
    

    trainDataset, valDataset = loadData(TrainPath, batchSize=config.batchSize , typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=config.batchSize)
    valDataset = valDataset.prefetch(buffer_size=config.batchSize)
    model = buildModel(InputShape)
    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.fit(trainDataset, epochs = config.epochs)

#run()

wandb.agent(sweep_id,function=run)