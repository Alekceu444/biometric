import cv2 as cv
import numpy as np
from scipy.fftpack import dct
from skimage.transform import rescale
class BC:
    def __init__(self, hist_size=256):
        self.hist_size = hist_size
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.train_histograms = []
        for train_image in self.X_train:
            train_hist = cv.calcHist([train_image], [0], None, [self.hist_size],[0,256])
            self.train_histograms.append(train_hist)
    def predict(self, test_images):
        predictions = []
        for test_image in test_images:
            distances = []
            test_hist = cv.calcHist([test_image], [0], None, [self.hist_size],[0,256])
            for train_hist in self.train_histograms:
                dist = np.linalg.norm(test_hist - train_hist)
                distances.append(dist)
            prediction = self.y_train[np.argmin(distances)]
            predictions.append(prediction)
        return np.array(predictions)
class DFTC:
    def __init__(self, size=1.0):
        self.size = size
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.max_row = int(X_train[0].shape[0]*self.size)
        self.max_column = int(X_train[0].shape[1]*self.size)
        self.train_dfts = []
        for train_image in self.X_train:
            train_dft = np.fft.fft2(train_image)
            train_dft = np.fft.fftshift(train_dft)
            train_dft = 20*np.log(np.abs(train_dft))
            train_dft = train_dft[:self.max_row, :self.max_column]    
            self.train_dfts.append(train_dft)
    def predict(self, test_images):
        predictions = []
        for test_image in test_images:
            distances = []
            test_dft = np.fft.fft2(test_image)
            test_dft = np.fft.fftshift(test_dft)
            test_dft = 20*np.log(np.abs(test_dft))
            test_dft = test_dft[:self.max_row, :self.max_column]
            for train_dft in self.train_dfts:
                dist = np.linalg.norm(test_dft.ravel() - train_dft.ravel())
                distances.append(dist)
            prediction = self.y_train[np.argmin(distances)]
            predictions.append(prediction)
        return np.array(predictions)
class DCTC:
    def __init__(self, size=1.0):
        self.size = size
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.max_row = int(X_train[0].shape[0]*self.size)
        self.max_column = int(X_train[0].shape[1]*self.size)
        self.train_dcts = []
        for train_image in self.X_train:
            train_dct = dct(dct(train_image.T, norm='ortho').T, norm='ortho')
            train_dct = train_dct[:self.max_row, :self.max_column]
            self.train_dcts.append(train_dct)    
    def predict(self, test_images):
        predictions = []
        for test_image in test_images:
            distances = []
            test_dct = dct(dct(test_image.T, norm='ortho').T, norm='ortho')
            test_dct = test_dct[:self.max_row, :self.max_column]
            for train_dct in self.train_dcts:
                dist = np.linalg.norm(train_dct.ravel() - test_dct.ravel())
                distances.append(dist)
            prediction = self.y_train[np.argmin(distances)]
            predictions.append(prediction)
        return np.array(predictions)
class SC:
    def __init__(self, sizes=[1.0]):
        self.sizes = sizes
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.train_rescaled_imgs = []
        for train_image in self.X_train:
            rescaled = self.get_rescaled_imgs(train_image)
            self.train_rescaled_imgs.append(rescaled)
    def get_rescaled_imgs(self, img):
        rescaled_imgs = []
        for size in self.sizes:   
            rescaled_img = rescale(img, size, mode='constant', anti_aliasing=False, multichannel=False)
            rescaled_imgs.append(rescaled_img)
        return rescaled_imgs 
    def get_avg_distance(self, rescaled_img1, rescaled_img2):
        total_dist = 0
        for i in range(len(self.sizes)):
            total_dist += np.linalg.norm(rescaled_img1[i].ravel() - rescaled_img2[i].ravel()) 
        return total_dist / len(self.sizes)
    def predict(self, test_images):
        predictions = []
        for test_image in test_images:
            distances = []
            test_rescaled_img = self.get_rescaled_imgs(test_image)
            for train_rescaled_img in self.train_rescaled_imgs:
                dist = self.get_avg_distance(test_rescaled_img, train_rescaled_img)
                distances.append(dist)
            prediction = self.y_train[np.argmin(distances)]
            predictions.append(prediction) 
        return np.array(predictions)
class GC:
    def __init__(self, ksize=-1):
        self.ksize = ksize
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train= y_train
        self.train_gradients = []
        for train_image in self.X_train:
            train_grad = cv.Sobel(train_image, cv.CV_64F, 0, 1, ksize=self.ksize)
            self.train_gradients.append(train_grad)
    def predict(self, test_images):
        predictions = []
        for test_image in test_images:
            distances = []
            test_grad = cv.Sobel(test_image, cv.CV_64F, 0, 1, ksize=self.ksize)
            for train_grad in self.train_gradients:
                dist = np.linalg.norm(test_grad.ravel() - train_grad.ravel())
                distances.append(dist)
            prediction = self.y_train[np.argmin(distances)]
            predictions.append(prediction)
        return np.array(predictions)
class VC:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
    def fit(self, X_train, y_train):
        for estimator in self.estimators:
            estimator.fit(X_train, y_train)
    def predict(self, X_test):
        predictions = []
        for estimator in self.estimators:
            y_pred = estimator.predict(X_test)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=40), 0, predictions).argmax(axis=0)
        return predictions