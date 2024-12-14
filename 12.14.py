from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
(trainX, trainy), (testX, testy) = mnist.load_data()
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' % (trainX.mean(), trainX.std(),
testX.mean(), testX.std()))
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(trainX)
print('Data Generator mean=%.3f, std=%.3f' % (datagen.mean, datagen.std))
iterator = datagen.flow(trainX, trainy, batch_size=64)
batchX, batchy = iterator.next()
print(batchX.shape, batchX.mean(), batchX.std())
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
batchX, batchy = iterator.next()
print(batchX.shape, batchX.mean(), batchX.std())