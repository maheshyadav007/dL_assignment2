import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import wandb
from wandb.keras import WandbCallback
import socket

#from tensorflow.keras.applications.Xception import Xception

#Global Variables
if socket.gethostname() == "DESKTOP-ROKMKKK":
    TrainPath = r"E:\inaturalist_12K\train"
    TestPath = r"E:\inaturalist_12K\test" 
else:
   
    drive.mount('/content/drive')
    TrainPath = r"/content/drive/MyDrive/inaturalist_12K/train"
    TestPath = r"/content/drive/MyDrive/inaturalist_12K/test"

ImageSize = (256,256)
InputShape = (256,256,3)
nClassifiers = 10

def loadData(path, batchSize = 64, typeData = None):
    if typeData == "train":
        trainDataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "training",  batch_size=batchSize, image_size = ImageSize)
        valDataset   = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "validation",  batch_size=batchSize, image_size = ImageSize)
        return trainDataset, valDataset
    elif typeData == "test":
        dataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337,batch_size=batchSize, image_size = ImageSize)
        return dataset

def fetchPretrainedModel(modelName):
    if modelName == "Xception":
        preProcess = keras.applications.xception.preprocess_input
        baseModel = keras.applications.Xception(weights='imagenet', input_shape=InputShape, include_top=False)
    elif modelName == "InceptionV3":
        preProcess = keras.applications.inception_v3.preprocess_input
        baseModel = keras.applications.InceptionV3(weights='imagenet', input_shape=InputShape, include_top=False)
    elif modelName == "InceptionResNetV2":
        preProcess = keras.applications.inception_resnet_v2.preprocess_input
        baseModel = keras.applications.InceptionResNetV2(weights='imagenet', input_shape=InputShape, include_top=False)
    elif modelName == "ResNet50":
        preProcess = keras.applications.resnet50.preprocess_input
        baseModel = keras.applications.ResNet50(weights='imagenet', input_shape=InputShape, include_top=False)
    elif modelName == "NASNetLarge":
        preProcess = keras.applications.nasnet.preprocess_input
        baseModel = keras.applications.NASNetLarge(weights='imagenet', input_shape=InputShape, include_top=False)
    
    return baseModel, preProcess


def buildModel(modelName,dropout,_globalLayer,dataAugment = True,sizeFCHL=[]):
    '''
    #NOTE
    modelName : Name of the pretrained model to be used
    dataAugment : Boolean to denote whether to use data augmentation
    globalLayer : which technique to use to convert feature map to flatten layer
    sizeFCHL : array that denotes the size of each fully connected hidden dense layer after base layer
    dropout : dropout to be used
    '''

    baseModel, modelPreProcessor = fetchPretrainedModel(modelName)
    baseModel.trainable = False

    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        ])

    if _globalLayer == "GlobalAveragePooling2D":
        globalLayer = keras.layers.GlobalAveragePooling2D()
    elif _globalLayer == "GlobalMaxPool2D":
        globalLayer = keras.layers.GlobalMaxPool2D()
    
    predictionLayer = keras.layers.Dense(nClassifiers, activation = "softmax")
    
    #Stack all layers
    inputLayer = keras.Input(shape=InputShape)
    if dataAugment:
        x = data_augmentation(inputLayer)
    x = modelPreProcessor(x)
    x = baseModel(x, training=False)
    x = globalLayer(x)
    x = layers.Dropout(dropout)(x)
    for units in sizeFCHL:
        x = keras.layers.Dense(units,activation="relu")(x)
        x = layers.Dropout(dropout)(x)

    output = predictionLayer(x)
    model = keras.Model(inputLayer, output)
    return model

def fineTune(model,k):
    
    model.layers[1].trainable = True

    for layer in model.layers[1].layers[0:-k]:
        layer.trainable = False

    return model
    

def getCallbacks(isWandBActive):
    callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    if isWandBActive:
        callbacks = [WandbCallback(),callback]
    else:
        callbacks = [callback]
    return callbacks

def getOptimizer(optimizerName,learningRate):
    if optimizerName == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    elif optimizerName == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learningRate)
    elif optimizerName == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    return optimizer

def getDataset(batchSize):
    trainDataset, valDataset = loadData(TrainPath, batchSize, typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=batchSize)
    valDataset = valDataset.prefetch(buffer_size=batchSize)
    return trainDataset, valDataset

def run(isWandBActive):
    if isWandBActive:
        config = wandb.config
        batchSize = config.batch_size
        dropout = config.dropout
        activation = config.activation
        fineTuneDepth = config.fine_tune_depth
        baseModel = config.base_model
        optimizerName = config.optimizer
        gFL = config.global_flattening_layer
        denseLayerDepth = config.dense_layer_depth
        learningRate = config.learning_rate
    else:
        batchSize = 64
        dropout = 0.2
        activation = "relu"
        fineTuneDepth = 8
        baseModel = "InceptionResNetV2"
        optimizerName = "Adam"
        gFL = "GlobalAveragePooling2D"
        denseLayerDepth = 0
        learningRate = 1e-3

    initial_epochs = 10
    fine_tune_epochs = 10
    sizeFCHL = [128]*denseLayerDepth

    
    optimizer = getOptimizer(optimizerName,learningRate)
    callbacks = getCallbacks(isWandBActive)
    trainDataset, valDataset = getDataset(batchSize)
    

    model = buildModel(modelName=baseModel,dropout=dropout,_globalLayer=gFL,dataAugment=True,sizeFCHL=sizeFCHL)
    model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    history = model.fit(trainDataset, epochs = initial_epochs,validation_data=valDataset,callbacks=callbacks)

    #Fine tuning
    total_epochs =  initial_epochs + fine_tune_epochs
    model = fineTune(model, fineTuneDepth)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    history_fine = model.fit(trainDataset,epochs=total_epochs,initial_epoch=history.epoch[-1],validation_data=valDataset,callbacks=callbacks)


sweep_config = {
      'method' : 'random',
      'metric' : {
          'name' : 'accuracy',
          'goal' : 'maximize'
      },
      'parameters' : {
          'learning_rate' : {'values' : [1e-2, 1e-3,1e-4]},
          'batch_size' : {'values' : [32, 64, 128]},
          'dense_layer_depth' : {'values' : [0,1]},
          'epochs' : {'values' : [10]},
          'dropout' : {'values' : [0.2,0.4, 0.5]},
          'activation' : {'values' : ['sigmoid', 'tanh', 'relu']},
          'base_model' : {'values' : ['Xception', 'InceptionV3', 'InceptionResNetV2','ResNet50']},
          'fine_tune_depth':{'values':[5,8,10,15]},
          'optimizer':{'values' : ['Adam']},
          'global_flattening_layer':{'values' : ['GlobalAveragePooling2D','GlobalMaxPool2D']},
      }
      
}
isWandBActive = True

if isWandBActive:
    sweep_id = wandb.sweep(sweep_config, entity="dl_assignment2", project="ConvolutionalNN")
    wandb.init()
    wandb.agent(sweep_id,function=run(isWandBActive))
else:
    run(isWandBActive)