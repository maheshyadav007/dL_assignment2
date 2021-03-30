import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential


inputShape = (1000,1000,3)

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
        

    model.add(layers.Dense(nDense, activation="relu"))
    model.add(layers.Dense(nClassifiers, activation = "softmax"))

    return model
#model.summary()
model = buildModel()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_valid, y_valid),callbacks=[checkpointer])