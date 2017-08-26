import preprocessData as preprocess
from keras.layers import LSTM, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential




def buildModel(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


print("Loading Data")
xTrain, yTrain, xTest, yTest = preprocess.preProcessData()

model = buildModel((None, 74))
model.fit(xTrain, yTrain, epochs=100, batch_size=64)

scores = model.evaluate(xTest, yTest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
