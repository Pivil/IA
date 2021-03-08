class DataManagement:
    from mlxtend.data import loadlocal_mnist
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


    trainImages, trainLabels = loadlocal_mnist(
        images_path='./samples/train-images-idx3-ubyte', 
        labels_path='./samples/train-labels-idx1-ubyte')
    testImages, testLabels = loadlocal_mnist(
        images_path='./samples/t10k-images-idx3-ubyte', 
        labels_path='./samples/t10k-labels-idx1-ubyte')

    def getDigitDistribution(self):
        return self.np.bincount(self.trainLabels)

    def displayImage(self, n):
        fig = self.plt.figure
        image = self.np.reshape(self.trainImages[n], (-1, 28))
        self.plt.imshow(image, cmap='gray')
        self.plt.show()

    def getMeanImage(self, label):
        #Get images that matche label
        imagesMatchingLabel = []
        for i in range(len(self.trainLabels)):
            if (self.trainLabels[i] == label):
                imagesMatchingLabel.append(i)
      
        #Get total value for pixel i for each image
        totalPixels = []
        for i in range(28*28):
            tmp = []
            for nbImage in imagesMatchingLabel:
                tmp.append(self.trainImages[nbImage, i])
            totalPixels.append(tmp)

        avgImg = []
        for i in range(len(totalPixels)):
            avgImg.append(self.np.mean(totalPixels[i]))
        
        fig = self.plt.figure
        image = self.np.reshape(avgImg, (-1, 28))
        return avgImg
    
    def getAllDataForDigit(self, digit):
        res = []
        nb = 0
        limit = 100
        for i in range(len(self.trainLabels)):
            if (self.trainLabels[i] == digit):
                if (nb < limit):
                    res.append(self.trainImages[i])
                    nb = nb + 1
        return res

    def showChart(self):
        fig = self.go.Figure(data=self.go.Bar(y=self.getDigitDistribution()))
        fig.show()
    
    def test(self, prediction, i):
        return 1/m * np.sum(prediction - self.getMeanImage(i))
            
