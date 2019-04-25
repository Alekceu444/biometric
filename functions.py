import cv2 as cv
import numpy as np
from Classifiers import *
import matplotlib.pyplot as plt
def accuracy_score(y_pred, y_test):
    return sum(y_pred == y_test) / len(y_test)
def plot_accuracy(test_sizes, accuracies, clf_name):
    plt.plot(test_sizes, accuracies)
    plt.xlabel('параметр')
    plt.ylabel('accuracy')
    plt.title(clf_name);
def acc_classifier(classifier,X_train, y_train):
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(accuracy_score(y_pred, y_train))
def check(pred,true,image,images):
    if true == pred:
        plt.imshow(image, cmap="gray")
        return(0)
    else:
        plt.imshow(images[pred], cmap="gray")
        return(1)
def checkOr(k, image):
    if(k==1):
        plt.imshow(image, cmap="hot")
        plt.title("Оригинал")
    else:
        plt.imshow(image, cmap="gray")
        plt.title("Оригинал")
def all_classifiers(classifier,X_train,y_train,X_test):
    clf = classifier    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred
def vote(X_train, y_train,X_test, y_test, bc=256,dftc=1,dctc=1,sc=[1],gc=-1):
    estimators = [BC(bc), DFTC(dftc), DCTC(dctc), 
              SC(sc), GC(gc)]
    voting = VC(estimators=estimators)
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    return accuracy_score(y_pred, y_test),y_pred,y_test
