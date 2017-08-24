import preprocessData as preprocess
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers import Dense


def buildModel(layers):
    model = Sequential()

    model.add(Conv2D)


xTrain, yTrain, xTest, yTest = preprocess.preProcessData()
