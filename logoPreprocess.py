import cv2
import sys
import numpy as np
import time
from regions import *
import copy
import os


def preprocess(image_path, show=False):

    image = cv2.imread(image_path)
    print (image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale

    filter_width = 3
    filter_height = gray.shape[0] // 20

    bkgrnd_removed_main = remove_background(gray, show, (filter_height, filter_width))

    logoFound = False
    ITER = 0            # dilation + erosion iterations
    BRT = 1.5           # main brightening factor

    while not logoFound:
        bkgrnd_removed = copy.deepcopy(bkgrnd_removed_main)
        if ITER == 4:
            BRT = 1
            ITER = 2
        height = bkgrnd_removed.shape[0]
        upper = np.int(height*0.4)
        brightening_factor = np.mean(bkgrnd_removed[upper:height-1, :])/np.mean(bkgrnd_removed[:upper, :])
        print("Brightening factor: %s" % (brightening_factor))
        brightening_factor *= BRT

        bkgrnd_removed = np.where(bkgrnd_removed*brightening_factor > 255, 255, bkgrnd_removed*brightening_factor)
        #take upper half of picture; logo cannot be lower than half
        bkgrnd_removed = bkgrnd_removed[:np.int(0.5 * bkgrnd_removed.shape[0]), :].astype(np.uint8)

        cv2.imshow('posvijetljena', bkgrnd_removed)

        ret, thresh = cv2.threshold(bkgrnd_removed,0,255,cv2.THRESH_OTSU) #threshold
        bin = bkgrnd_removed < ret                      #binary (problem)
        thresh = 255 - thresh
        print ret, thresh
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        dilated = cv2.dilate(thresh,kernel,iterations = ITER)  # dilate
        eroded = cv2.erode(dilated,kernel,iterations = ITER)   # erode


        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # get contours
        # for each contour found, draw a rectangle around it on original image
        print("Found: " + str(len(contours)) + " contours.")
        final = copy.deepcopy(bkgrnd_removed)

        rectangles = list()
        for contour in contours:
            # get rectangle bounding contour

            [x,y,w,h] = cv2.boundingRect(contour)
            # discard areas that are too large

            if h>0.4*image.shape[0] or w>0.8*image.shape[1]:

                continue
            # discard areas that are too small

            if h<10 or w<10:

                continue

            if h/w > 3:
                continue

            x1 = y  #swithed x and y
            y1 = x
            x2 = x1 + h
            y2 = y1 + w

            if x1 <= 1 or y1 <= 1 or y2 >= final.shape[1] -1 or x2 >= final.shape[0] -1:
                continue


            # draw rectangle around contour on original image
            rectangles.append((x,y,w,h))
            cv2.rectangle(final,(x,y),(x+w,y+h),(0,0,255),2)

        bbox = logo_box(final, rectangles, show)

        print(bbox)
        if bbox == 0:
            logoFound = False
            logo = np.zeros((60, 100))
            ITER += 1

        else:
            logoFound = True
            logo = thresh[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]


    cv2.imshow('LOGO', logo)

    while logo.shape[0] < 60:

        logo = np.vstack((np.zeros(logo.shape[1])[np.newaxis, ...], logo))
        logo = np.vstack((logo, np.zeros(logo.shape[1])[np.newaxis, ...]))


        #print("Premali logo")


    while logo.shape[1] < 100:
        logo = np.hstack((np.zeros(logo.shape[0])[..., np.newaxis], logo))
        logo = np.hstack((logo, np.zeros(logo.shape[0])[..., np.newaxis]))



    if show:
        cv2.imshow("grayscale", gray)
        cv2.imshow("background removed", bkgrnd_removed)
        cv2.imshow("thresholded", thresh)
        cv2.imshow("eroded+dilated", eroded)
        cv2.imshow("final", final)
        cv2.waitKey(0)






    return logo


def siftPreprocess(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
    cv2.imshow("grayscale", gray)
    cv2.waitKey(0)

    _,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV) #threshold
    cv2.imshow("thresholded", thresh)
    cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 3) # dilate
    cv2.imshow("dilated", dilated)
    cv2.waitKey(0)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    print(kp)
    image = cv2.drawKeypoints(gray,kp,image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("keypoints", image)
    cv2.waitKey(0)

    #cv2.imwrite('sift_keypoints.jpg',img)



def crossCorr(img1, img2):
    if (img1.shape != img2.shape):
        print("Images don't have the same shape.")
        return 0
    corrs = np.array([])
    h = img1.shape[0]
    w = img1.shape[1]
    limitW = 2*w - 1
    limitH = 2*h - 1
    corrs = np.append(corrs, np.sum(1 - np.bitwise_xor(img1, img2)))
    print(corrs)
    for i in range(2, w - 2):
        corr = np.sum(1 - np.bitwise_xor(img1[: , i:], img2[: , :w-i]))
        corrs = np.append(corrs, corr)

        corr = np.sum(1 - np.bitwise_xor(img1[: , :w-i], img2[: , i:]))
        corrs = np.append(corrs, corr)

    for i in range(2, h - 2):
        corr = np.sum(1 - np.bitwise_xor(img1[i: , :], img2[:h-i , :]))
        corrs = np.append(corrs, corr)

        corr = np.sum(1 - np.bitwise_xor(img1[:h-i , :], img2[i: , :]))
        corrs = np.append(corrs, corr)

    return np.max(corrs)

if __name__ == "__main__":
    start_time = time.time()
    current_dir = os.getcwd()
    preprocess(sys.argv[1], sys.argv[2])

    #img1 = cv2.imread(sys.argv[1], 0)
    #img2 = cv2.imread(sys.argv[2], 0)
    #img1 = img1 > 128
    #img2 = img2 > 128
    #print(crossCorr(img1, img2))




    print("--- %s seconds ---" % (time.time() - start_time))
