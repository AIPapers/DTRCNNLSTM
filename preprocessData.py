import glob
import os

import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

import prepareNumpyArray as prepare


def preProcessData():
    root = '/home/striker/PycharmProjects/DTRCNNLSTM/video_files/'
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

    print(dirlist)

    labels = []
    dfImages = []
    totalLabelNumbers = []
    numberOfVideos = []
    maxSeqLength = 0

    for i in range(len(dirlist)):
        vids = []
        nums = 0
        print(root + dirlist[i] + "/*.avi")
        labels.append(int(dirlist[i][dirlist[i].find("_") + 1:]))
        for vid in glob.glob(root + dirlist[i] + '/*.avi'):
            vids.append(vid)

        numberOfVideos.append(len(vids))
        for i in range(len(vids)):
            dfImage, maxL, num = prepare.getImages(vids[i])
            for x in dfImage:
                dfImages.append(np.array(x))
            nums += num
            maxSeqLength = max(maxSeqLength, maxL)
        totalLabelNumbers.append(nums)

    dfImages = np.dstack(dfImages)
    targetValues = np.zeros(sum(totalLabelNumbers))

    count = 0
    for i in range(len(labels)):
        targetValues[count:count + totalLabelNumbers[i]] = labels[i]
        count += totalLabelNumbers[i]

    print(len(dfImages), len(targetValues))
    print(dfImages[0].shape, dfImages.shape)


    dfImagesFlat = np.array([dfImages[:, :, i] for i in range(len(dfImages[0, 0, :]))])
    targetValues = np.array(targetValues)
    print(dfImagesFlat.shape, targetValues.shape)
    pyplot.imshow(dfImagesFlat[0])

    x_train, x_test, y_train, y_test = train_test_split(dfImagesFlat, targetValues)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.reshape(x_train.shape[0], 1, 74, 74)
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], 1, 74, 74)
    x_test /= 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(x_test.shape, y_test.shape)

    # print(x_test.head(), y_test.head())
    print(len(x_test), len(x_train), maxSeqLength)

    return x_train, y_train, x_test, y_test
