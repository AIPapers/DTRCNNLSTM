import cv2

import capVideoFrames as cap


def resizeImage(img):
    targetLength = targetWidth = 75
    top = bottom = left = right = 0
    imgLength = len(img)
    imgWidth = len(img[0])

    difLength = targetLength - imgLength
    if (difLength % 2 == 0):
        top = bottom = difLength // 2
    else:
        top = difLength // 2 + 1
        bottom = difLength // 2

    difWidth = targetWidth - imgWidth
    if (difWidth % 2 == 0):
        left = right = difWidth // 2
    else:
        left = difWidth // 2 + 1
        right = difWidth // 2

    return cv2.resize(
        cv2.copyMakeBorder(img, top=top, left=left, right=right, bottom=bottom, borderType=cv2.BORDER_CONSTANT),
        (64, 64))


def getImages(ims):
    image, count = cap.readImages(ims)
    x, y, z = image.shape
    dfImage = []

    for i in range(x):
        hsimage = (image[i, :, :])
        dfImage.append(resizeImage(hsimage))

    for i in range(y):
        hsimage = (image[:, i, :])
        dfImage.append(resizeImage(hsimage))

    for i in range(z):
        hsimage = (image[:, :, i])
        dfImage.append(resizeImage(hsimage))

    del image, count

    return dfImage, max(x, y, z), (x + y + z)
