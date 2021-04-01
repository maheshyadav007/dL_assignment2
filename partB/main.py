import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
#from tensorflow.keras.applications.Xception import Xception

#Global Variables
TrainPath = r"E:\inaturalist_12K\train"
TestPath = r"E:\inaturalist_12K\test" 
ImageSize = (256,256)
InputShape = (256,256,3)
nClassifiers = 10

def loadData(path, batchSize = 64, typeData = None):
    scaler = Rescaling(scale=1.0 / 255)
    if typeData == "train":
        trainDataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "training",  batch_size=batchSize, image_size = ImageSize)
        valDataset   = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "validation",  batch_size=batchSize, image_size = ImageSize)
        return trainDataset, valDataset
    elif typeData == "test":
        dataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337,batch_size=batchSize, image_size = ImageSize)
        return dataset

def buildModel():
    baseModel = keras.applications.Xception(weights='imagenet', input_shape=InputShape, include_top=False)
    for layer in baseModel.layers:
      layer.trainable = False

    inputLayer = keras.Input(shape=InputShape)
    x = baseModel(inputLayer, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(nClassifiers, activation = "softmax")(x)
    model = keras.Model(inputLayer, output)
    return model

def fineTune(model,k):
    length = len( model.layers[1].layers)

    for layer in model.layers[1].layers[0:length-k]:
        layer.trainable = False
    for layer in model.layers[1].layers[length-k:length-1]:
        layer.trainable = True
    
    return model
    

def run():
    
    batchSize = 128
    epochs = 10
    k = 4

    trainDataset, valDataset = loadData(TrainPath, batchSize, typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=batchSize)
    valDataset = valDataset.prefetch(buffer_size=batchSize)
    
    model = buildModel()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.fit(trainDataset, epochs = epochs)

    #Fine tuning
    model = fineTune(model, k)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.fit(trainDataset, epochs = epochs)

run()