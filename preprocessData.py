import glob
import os
import numpy as np
from sklearn.utils import shuffle
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

    x_train, x_test, y_train, y_test = train_test_split(dfImagesFlat, targetValues)

    print(x_test[0:2], y_test[0:2])

    # print(x_test.head(), y_test.head())
    print(len(x_test), len(x_train), maxSeqLength)

    return x_train, y_train, x_test, y_test
