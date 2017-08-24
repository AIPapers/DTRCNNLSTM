import cv2 as cv
import numpy as np
import skimage.measure as m
import skvideo.io as skv


def readImages(ims, thresholdParam=0.9):
    images = skv.vread(ims)
    count = 0
    temp = images[0]
    temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

    numberOfFrames, a, b, c = images.shape

    finalImages = np.zeros((numberOfFrames, a, b))
    finalImages[0] = temp

    d = []

    for image in images:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        score = m.compare_ssim(temp, image)
        d.append(score)

        if score < thresholdParam:
            finalImages[count] = image
            temp = image
            count += 1

    finalImages = finalImages[:count]
    return finalImages, count
