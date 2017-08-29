import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

import prepareNumpyArray as prepare


def preProcessData():
    root = '/home/striker/PycharmProjects/DTRCNNLSTM/video_files/'
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

    labels = []
    dfImages = []
    totalLabelNumbers = []
    maxSeqLength = 0

    for i in range(len(dirlist)):
        vids = []
        nums = 0
        labels.append(int(dirlist[i][dirlist[i].find("_") + 1:]))
        for vid in glob.glob(root + dirlist[i] + '/*.avi'):
            vids.append(vid)

        for i in range(len(vids)):
            dfImage, maxL, num = prepare.getImages(vids[i])
            for x in dfImage:
                dfImages.append(np.array(x))
            nums += num
            maxSeqLength = max(maxSeqLength, maxL)
        totalLabelNumbers.append(nums)

    del dirlist
    dfImages = np.dstack(dfImages)
    targetValues = np.zeros(sum(totalLabelNumbers))

    count = 0
    for i in range(len(labels)):
        targetValues[count:count + totalLabelNumbers[i]] = labels[i]
        count += totalLabelNumbers[i]


    dfImagesFlat = np.array([dfImages[:, :, i] for i in range(len(dfImages[0, 0, :]))])

    del dfImages, totalLabelNumbers, labels, maxSeqLength

    targetValues = np.array(targetValues)

    x_train, x_test, y_train, y_test = train_test_split(dfImagesFlat, targetValues)

    del dfImagesFlat, targetValues

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)
    x_test /= 255

    return x_train, y_train, x_test, y_test
