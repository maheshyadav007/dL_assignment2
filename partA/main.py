import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


#Global Variables
TrainPath = r"E:\inaturalist_12K\train"
TestPath = r"E:\inaturalist_12K\test" 
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
    scaler = Rescaling(scale=1.0 / 255)
    if typeData == "train":
        trainDataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "training",  batch_size=batchSize, image_size = ImageSize)
        valDataset   = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337, validation_split = 0.1, subset = "validation",  batch_size=batchSize, image_size = ImageSize)
        return trainDataset, valDataset
    elif typeData == "test":
        dataset = keras.preprocessing.image_dataset_from_directory(path, label_mode = 'categorical',seed=1337,batch_size=batchSize, image_size = ImageSize)
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
        model.add(layers.Dense(nDense[i], activation="relu"))

    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model


#model.add(layers.GlobalMaxPooling2D())
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_valid, y_valid),callbacks=[checkpointer])

def run():
    
    batchSize = 64
    epochs = 10

    trainDataset, valDataset = loadData(TrainPath, batchSize, typeData = "train")
    trainDataset = trainDataset.prefetch(buffer_size=batchSize)
    valDataset = valDataset.prefetch(buffer_size=batchSize)
    model = buildModel(InputShape)
    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.fit(trainDataset, epochs = epochs)

run()