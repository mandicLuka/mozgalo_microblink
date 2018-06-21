import os
import sys
import random
import numpy as np
import cv2
from math import ceil

def load_images(trainDir, batchSize=-1, startFrom=0):
    """
    Input:

        -trainDir = full path to training directory,
                    including the directory itself
        -batchSize = number of total images to load
        -startFrom = index of first image to load from each class folder

    Output:

        if batchSize = -1 outputs testSet
        else outputs trainSet and its labels
    """


    trainSet = []
    testSet = []
    labels = []

    if not os.path.isdir(trainDir):
        sys.exit("Target path isn't a directory.")


    allLabels = os.listdir(trainDir)
    allLabels.sort()            #sort alphabetically in order

    return_labels = 0

    if not allLabels:
        sys.exit("No training data.")


    samplesPerClass = np.int(ceil(batchSize / 25)) #25 classes available
    

    #TRAIN SET LOADING
    if batchSize > 0 :
        print("Taking " + str(samplesPerClass) + " samples per class.")

        classCounter = 0
        while (len(labels) < batchSize):
            currPath = trainDir + "/" + allLabels[classCounter] #create path to current class
            #print("Currently reading " + currPath)
            currLabel = os.listdir(currPath)
            currLabel.sort()
            if startFrom + samplesPerClass >= len(currLabel):
                currSize = range(startFrom, len(currLabel))
            else:
                currSize = range(startFrom, startFrom + samplesPerClass)

            for j in currSize:
                if not currLabel[j].endswith(".jpg"):
                    print("File " + currLabel[j] + " isn't a .jpg!")
                    continue
                trainSet.append(cv2.imread(currPath + "/" + currLabel[j]))    #read current image
                temp = np.zeros(25)
                temp[classCounter] = 1              #create one-hot vector for the image
                labels.append(temp)

            classCounter += 1
            if classCounter == 25:
                break

        return trainSet, labels


    #TEST SET LOADING
    else:
        batchSize = 100
        print("Taking " + str(batchSize) + " samples.")

        samplesPerClass = batchSize
        currPath = trainDir + "/" + allLabels[0]
        currLabel = os.listdir(currPath)
        currSize = []
        if startFrom + samplesPerClass >= len(currLabel):
            currSize = range(startFrom, len(currLabel))
        else:
            currSize = range(startFrom, startFrom + samplesPerClass)
        for j in currSize:
            if not currLabel[j].endswith(".jpg"):
                print("File " + currLabel[startFrom + j] + " isn't a .jpg!")
                continue
            testSet.append(cv2.imread(currPath + "/" + str(j)+ ".jpg"))    #read current image
        
        del currSize, currLabel
        return testSet
