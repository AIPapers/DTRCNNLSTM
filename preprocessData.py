import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import prepareNumpyArray as prepare


def preProcessData():
    root = '/home/striker/PycharmProjects/DTRCNN/data/videoFiles/'
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
        print(vids)
        numberOfVideos.append(len(vids))
        for i in range(len(vids)):
            dfImageXY, dfImageXZ, dfImageYZ, maxL = prepare.getImages(vids[i])
            tempXY = [x for x in dfImageXY]
            tempXZ = [x for x in dfImageXZ]
            tempYZ = [x for x in dfImageYZ]
            dfImages.append(tempXY)
            dfImages.append(tempXZ)
            dfImages.append(tempYZ)
            maxSeqLength = max(maxSeqLength, maxL)
        totalLabelNumbers.append(len(vids) * 3)

    dfImages = np.array(dfImages)
    targetValues = np.zeros(sum(numberOfVideos) * 3)

    count = 0
    for i in range(len(labels)):
        targetValues[count:count + totalLabelNumbers[i]] = labels[i]
        count += totalLabelNumbers[i]

    print(len(dfImages), len(targetValues))

    dfFinal = pd.DataFrame({'image': dfImages, 'label': targetValues})
    print(dfFinal.iloc[[35]])

    df_train, df_test = train_test_split(dfFinal)
    x_train = pd.DataFrame(df_train['image'])
    y_train = pd.DataFrame(df_train['label'])
    x_test = pd.DataFrame(df_test['image'])
    y_test = pd.DataFrame(df_test['label'])

    # print(x_test.head(), y_test.head())
    print(len(x_test), len(x_train), len(x_test.iloc[[1]]), maxSeqLength)

    return x_train, y_train, x_test, y_test
