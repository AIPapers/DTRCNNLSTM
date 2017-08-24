import capVideoFrames as cap


def getImages(ims):
    dfImage = []
    image, count = cap.readImages(ims)
    x, y, z = image.shape

    for i in range(x):
        hsimage = image[i, :, :]
        dfImage.append(hsimage)

    for i in range(y):
        hsimage = image[:, i, :]
        dfImage.append(hsimage)

    for i in range(z):
        hsimage = image[:, :, i]
        dfImage.append(hsimage)

    print(x)
    return dfImage, max(x, y, z), (x + y + z)
