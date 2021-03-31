import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras import Sequential

path = r"E:\inaturalist_12K\train"

batchSize = 64
dataset = keras.preprocessing.image_dataset_from_directory(path,label_mode = 'categorical', batch_size=batchSize)

'''
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels[0])  # int32
'''
inputShape = (256,256,3)
#inputShape = (inputShape[1],inputShape[2],inputShape[3])

def buildModel():
    model = Sequential()
    model.add(keras.Input(shape=inputShape))

    convLayerSize = 5
    nFilters = [32]*convLayerSize
    filterSize = [5]*convLayerSize
    activationFuncs = ["relu"]*convLayerSize
    maxPoolSize = [2]*convLayerSize
    nDense = 64
    nClassifiers = 10

    for i in range(convLayerSize):
        model.add(layers.Conv2D(nFilters[i], filterSize[i], strides=1, activation=activationFuncs[i]))
        model.add(layers.MaxPooling2D(maxPoolSize[i]))
        
      
    model.add(Flatten())
    model.add(layers.Dense(nDense, activation="relu"))
    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model

model = buildModel()
model.summary()
#model.add(layers.GlobalMaxPooling2D())
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_valid, y_valid),callbacks=[checkpointer])
model.fit(dataset, epochs=10)
