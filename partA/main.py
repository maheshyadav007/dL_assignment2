!pip install wandb
import wandb
from wandb.keras import WandbCallback
from google.colab import drive
import tensorflow as tf
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops
from tensorflow.keras.models import Model
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt




#Global Variables
drive.mount('/content/gdrive')
TrainPath = "/content/gdrive/MyDrive/inaturalist_12K/train"
TestPath = "/content/gdrive/MyDrive/inaturalist_12K/val"
classNames = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
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
    model.summary()
    return model




def getCallbacks(isWandBActive):
    #Early Stopping is monitor on validation accuracy for Train Dataset
    if Train_data:
        callback = EarlyStopping(monitor='val_accuracy', patience=2)

    #Early Stopping is monitor on accuracy for Test Dataset
    else:
        callback = EarlyStopping(monitor='accuracy', patience=2)

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
	        'dataAugment' : True,
          'batchNormalization' : True,
          'dropout' : 0.3,
          'seed' : 42,
          'nFilters' : 128,
          'filterSize' : 5,
          'optimizer':'Adam',
          'global_flattening_layer' : 'Flatten',
	        'filterArrangement' : 'doubling',
          'denseLayerSize' : 1,
          'sizeMaxpool' : 2,
          'nDense' : 64
        }


#sweep config for Train Dataset

sweep_config = {
    'method' : 'random',
    'metric' : {
        'name' : 'accuracy',
        'goal' : 'maximize'
        },
    'parameters' : {
        'nFilters' : {'values' : [32, 64, 128]},
        'filterSize' : {'values' : [5, 10]},
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
        'denseLayerSize' : {'values' : [1]},
        } 
    }
'''
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
              'sizeMaxpool' : {'values' : [2]},
              'batchSize' : {'values' : [16]},
              'nDense' :  {'values' : [64]},
              'dataAugment' :  {'values' : [True]},
              'batchNormalization' : {'values' : [True]},
              'denseLayerSize' : {'values' : [1]},
              'filterSize' : {'values' : [5]},
              'epochs' : {'values' : [10]},
              'optimizer':{'values' : ['Adam']},
              'dropout ' : {'values' : [0.3]},
              'activationFuncs' : {'values' : ['tanh']},
              'filterArrangement' : {'values' : ['doubling']},
              'global_flattening_layer':{'values' : ['Flatten']}
          }     
    }

'''
'''
-------------------------------------------------Train Functoion---------------------------------------------------------------------
isWandBActive : Boolean variable to denote searching for hyperparam if ..False  then evaluation on best model
'''
isWandBActive = False

def train():
    convLayerSize = 5
    denseLayerSize = 1
    nClassifiers = 10

    if isWandBActive:
        wandb.init(config = config_defaults)
        config = wandb.config
        epochs = config.epochs
        batchSize = config.batchSize
        learning_rate = config.learning_rate
        activationFuncs = [config.activationFuncs]*convLayerSize
        dropout = config.dropout
        seed = config.seed
        nFilters = [config.nFilters]*convLayerSize
        sizeFilters = [config.filterSize]*convLayerSize
        optimizer = config.optimizer
        global_flattening_layer = config.global_flattening_layer
        sizeDenseLayers = [config.nDense]*denseLayerSize
        dataAugment = config.dataAugment
        batchNormalization = config.batchNormalization
        sizeMaxpool = [config.sizeMaxpool]*convLayerSize
        filterArrangement = config.filterArrangement
        sizeDenseLayers = [config.nDense]*denseLayerSize

    else:
        # Best Model's Hyperparameters

        nFilters = [64]*convLayerSize
        sizeFilters = [7]*convLayerSize
        activationFuncs = ["relu"]*convLayerSize
        sizeMaxpool = [2]*convLayerSize
        sizeDenseLayers = [256]*denseLayerSize
        dataAugment = True
        batchNormalization = True
        epochs = 1
        batchSize = 64
        learning_rate = 0.0001
        dropout = 0.1
        seed = 42
        optimizer = "Adam"
        global_flattening_layer = "Flatten"
        filterArrangement ="equal"

    callbacks = getCallbacks(isWandBActive)
        
    trainDataset, valDataset = loadData(TrainPath, batchSize=batchSize , typeData = "train")
    testDataset = loadData(TestPath,batchSize=batchSize, typeData= "test")

    testDataset = testDataset.prefetch(buffer_size=batchSize)
    trainDataset = trainDataset.prefetch(buffer_size=batchSize)
    valDataset = valDataset.prefetch(buffer_size=batchSize)

    model = buildModel(sizeFilters, nFilters, activationFuncs, sizeMaxpool, sizeDenseLayers, dropout, global_flattening_layer, filterArrangement, batchNormalization, dataAugment)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(trainDataset, epochs = epochs, validation_data=valDataset, callbacks=callbacks)

    return model, testDataset

if isWandBActive:
    sweep_id = wandb.sweep(sweep_config, entity="dl_assignment2", project="ConvolutionNN")
    wandb.agent(sweep_id,function=train)
else:
    model, testDataset = train()


#-------------------------------------Evaluation on best Model---------------------------

def evaluateModelPerformance(model, testDataset):
  loss, acc = model.evaluate(testDataset)
  print("Best model's accuracy on Test Data : ",acc*100 ,"%")


def predictImages(predictions,testDataset):
  plt.figure(figsize=(20, 20))
  for images, labels in testDataset.take(1):
      for i in range(30):
          ax = plt.subplot(10, 3, i + 1)
          plt.imshow(images[i].numpy().astype("uint8"))
          plt.title(classNames[np.argmax(predictions[i], axis= 0)])
          plt.axis("off")

def visualizeFilters(model,testDataset):
  for layer in model.layers:
    if 'conv' in layer.name:
      #filters, biases = layer.get_weights()
      #print(layer.name, filters.shape)

      model = keras.Model(inputs=model.inputs, outputs=layer.output)
      feature_maps = model.predict(testDataset.take(1))
      break

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

evaluateModelPerformance(model, testDataset)
predictions = model.predict(testDataset.take(30))
predictImages(predictions,testDataset)
visualizeFilters(model,testDataset)

#---------------------------------------Guided Backprop----------------------------------------------------------------


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

class GuidedBackprop:
    def __init__(self,model, layerName=None,fI=0,p1=0,p2=0):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()
        self.fI=fI
        self.p1=p1
        self.p2=p2
        self.gbModel = self.build_guided_model()
        
        
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output[0][self.p1][self.p2][self.fI]]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guidedBackprop(self, images, upsample_size):
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)
        grads = tape.gradient(outputs, inputs)[0]
        saliency = cv2.resize(np.asarray(grads), upsample_size)
        return saliency

def getProcessedImage(testDataset):
  for images, labels in testDataset.take(1):
    scaler = Rescaling(scale=1.0 / 255)
    processedImage = scaler(images[0])
    processedImage = np.expand_dims(processedImage, axis=0)
    return processedImage

def deprocessImage(x):
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

processedImage = getProcessedImage(testDataset)
layerName = model.layers[14].name

plt.figure(figsize=(50, 50))
for i in range(10):
    plt.subplot(10, 1, i+1)
    guidedBP = GuidedBackprop(model=model,layerName=layerName,fI=i,p1=0,p2=0)
    gbCam = guidedBP.guidedBackprop(processedImage,(256,256))
    gbIm = deprocessImage(gbCam)
    gbIm = cv2.cvtColor(gbIm, cv2.COLOR_BGR2GRAY)
    plt.imshow(gbIm,cmap="gray")
    plt.axis("off")
    plt.show()

#--------------------------------------------------------------------------------------------------------------