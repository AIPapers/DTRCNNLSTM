import capVideoFrames as cap


def getImages(ims):
    dfImageXY = dfImageXZ = dfImageYZ = []
    image, count = cap.readImages(ims)
    x, y, z = image.shape

    for i in range(x):
        hsimage = image[i, :, :]
        dfImageXY.append(hsimage)

    for i in range(y):
        hsimage = image[:, i, :]
        dfImageXZ.append(hsimage)

    for i in range(z):
        hsimage = image[:, :, i]
        dfImageYZ.append(hsimage)

    return dfImageXY, dfImageXZ, dfImageYZ, max(x, y, z)
