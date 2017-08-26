import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
                dfImages.append(x)
            nums += num
            maxSeqLength = max(maxSeqLength, maxL)
        totalLabelNumbers.append(nums)

    dfImages = np.dstack(dfImages)
    targetValues = np.zeros(sum(totalLabelNumbers))

    count = 0
    for i in range(len(labels)):
        targetValues[count:count + totalLabelNumbers[i]] = labels[i]
        count += totalLabelNumbers[i]

    print(len(dfImages[0, 0, :]), len(targetValues))
    print(dfImages[0].shape, dfImages.shape)

    dfFinal = pd.DataFrame({'images': dfImages, 'label': targetValues})

    print(dfFinal.head())

    df_train, df_test = train_test_split(dfFinal)

    '''x_train = np.array(df_train[0])
    y_train = np.array(df_train[1])
    x_test = np.array(df_test[0])
    y_test = np.array(df_test[1])

    # print(x_test.head(), y_test.head())
    print(len(x_test), len(x_train), maxSeqLength)

    return x_train, y_train, x_test, y_test'''
