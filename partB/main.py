import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Normalization
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


'''
#NOTE

1. Image Resizing 
2. normalization
3. data aug
4. position of dropout layer


'''
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
        baseModel = keras.applications.NASNetLarge(weights='imagenet', input_shape=(331,331,3), include_top=False)
    elif modelName == "VGG19":
        preProcess = keras.applications.vgg19.preprocess_input
        baseModel = keras.applications.VGG19(weights='imagenet', input_shape=InputShape, include_top=False)
    elif modelName == "EfficientNetB7":
        preProcess = keras.applications.efficientnet.preprocess_input
        baseModel = keras.applications.EfficientNetB7(weights='imagenet', input_shape=InputShape, include_top=False)
    else:
        raise Exception("Invalid Base Model Name")
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
    if modelName == "NASNetLarge":
        ImageSize = (331,331)
    else:
        ImageSize = (256,256)

    baseModel, modelPreProcessor = fetchPretrainedModel(modelName)
    baseModel.trainable = False

    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        #layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.Resizing(ImageSize[0], ImageSize[0]),
        #layers.experimental.preprocessing.Normalization()
        ])

    if _globalLayer == "GlobalAveragePooling2D":
        globalLayer = keras.layers.GlobalAveragePooling2D()
    elif _globalLayer == "GlobalMaxPool2D":
        globalLayer = keras.layers.GlobalMaxPool2D()
    elif _globalLayer == "Flatten":
        globalLayer = keras.layers.Flatten()
    else:
         raise Exception("Invalid Layer Name") 
    
    predictionLayer = keras.layers.Dense(nClassifiers, activation = "softmax")
    
    #Stack all layers
    inputLayer = keras.Input(shape=InputShape)
    if dataAugment:
        x = data_augmentation(inputLayer)
    x = modelPreProcessor(x)
    x = baseModel(x, training=False)
    #x = layers.Dropout(dropout)(x)
    x = globalLayer(x)
    x = layers.Dropout(dropout)(x)
    for units in sizeFCHL:
        x = keras.layers.Dense(units,activation="relu")(x)
        x = layers.Dropout(dropout)(x)

    output = predictionLayer(x)
    model = keras.Model(inputLayer, output)
    return model

def fineTune(model,modelName,k):
    #LOok for layers no. of base model
    for layer in model.layers:
        if modelName in layer.name:
            layer.trainable = True

            for l in layer[0:-k]:
                l.trainable = False
            print(layer.summary())
            break
    return model
    

def getCallbacks(isWandBActive):
    callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
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

def getDataset(batchSize=64):
    trainDataset, valDataset = loadData(TrainPath, batchSize, typeData = "train")
    
    

    trainDataset = trainDataset.prefetch(buffer_size=batchSize)

    
    valDataset = valDataset.prefetch(buffer_size=batchSize)



    #normalizer = Normalization(axis=-1)
    #normalizer.adapt(trainDataset)
    #trainDataset = normalizer(trainDataset)

    #trainDataset = trainDataset.map(lambda x, y: (tf.image.resize(x, ImageSize), y))
    #valDataset = valDataset.map(lambda x, y: (tf.image.resize(x, ImageSize), y))

    #trainDataset = trainDataset.cache().batch(batchSize).prefetch(buffer_size=10)
    #valDataset = valDataset.cache().batch(batchSize).prefetch(buffer_size=10)
    #test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    #test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    return trainDataset, valDataset


sweep_config = {
      'method' : 'random',
      'metric' : {
          'name' : 'accuracy',
          'goal' : 'maximize'
      },
      'parameters' : {
          'learning_rate' : {'values' : [1e-3,1e-4]},
          'batch_size' : {'values' : [32, 64, 128]},
          'dense_layer_depth' : {'values' : [0,1,2]},
          'sizeFCHL' : {'values' : [128,512,1024]},
          'epochs' : {'values' : [10]},
          'dropout' : {'values' : [0.2,0.4, 0.5]},
          'activation' : {'values' : ['sigmoid', 'tanh', 'relu']},
          'base_model' : {'values' : ['EfficientNetB7','Xception', 'InceptionV3', 'InceptionResNetV2','ResNet50','VGG19','NASNetLarge']},
          'fine_tune_depth':{'values':[5,10, 15]},
          'optimizer':{'values' : ['Adam','RMSprop']},
          'global_flattening_layer':{'values' : ['GlobalAveragePooling2D','GlobalMaxPool2D','Flatten']},
      }
      
}
hyperparameters_defaults = {
          'learning_rate' :1e-3,
          'batch_size' : 64,
          'dense_layer_depth' : 1,
          'sizeFCHL' : 1024,
          'epochs' : 10,
          'dropout' : 0.2,
          'activation' : 'relu',
          'base_model' : 'InceptionResNetV2',
          'fine_tune_depth': 8,
          'optimizer':'Adam',
          'global_flattening_layer':'GlobalAveragePooling2D'
        }

        
'''
----------------------------------------------------------------------------------------------------------------------
isWandbActive : whether to run with or without wandb

NOTE:
1. Look for NASNetLarge

'''


isWandBActive = False
trainDataset, valDataset = getDataset()

def train():
    if isWandBActive:
        wandb.init(config = hyperparameters_defaults)
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
        sizeFCHL = [config.sizeFCHL]*denseLayerDepth

    else:
        batchSize = 64
        dropout = 0.4
        activation = "relu"
        fineTuneDepth = 10
        baseModel = "NASNetLarge"
        optimizerName = "Adam"
        gFL = "GlobalAveragePooling2D"
        denseLayerDepth = 1
        learningRate = 1e-3
        sizeFCHL = [1024]*denseLayerDepth

    initial_epochs = 10
    fine_tune_epochs = 10
    

    
    optimizer = getOptimizer(optimizerName,learningRate)
    callbacks = getCallbacks(isWandBActive)
    
    

    model = buildModel(modelName=baseModel,dropout=dropout,_globalLayer=gFL,dataAugment=True,sizeFCHL=sizeFCHL)
    model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    history = model.fit(trainDataset, epochs = initial_epochs,validation_data=valDataset,callbacks=callbacks)

    #Fine tuning
    total_epochs =  initial_epochs + fine_tune_epochs
    model = fineTune(model,baseModel, fineTuneDepth)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    history_fine = model.fit(trainDataset,epochs=total_epochs,initial_epoch=history.epoch[-1],validation_data=valDataset,callbacks=callbacks)



if isWandBActive:
    sweep_id = wandb.sweep(sweep_config, entity="dl_assignment2", project="ConvolutionalNN")
    wandb.agent(sweep_id,function=train)
else:
    train()