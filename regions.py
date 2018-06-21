from __future__ import (
    division,
    print_function,
)

from math import sqrt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import time

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import scipy.ndimage.filters as fi


class RectangleSet(mpatches.Rectangle):
    '''
        Rectangle class for gruping rectangles
    '''

    def __init__(self, rect):

        bbox = rect.get_bbox()
        self.rect_num = 1
        self.upper_left = (bbox.x0, bbox.y0)
        self.lower_right = (bbox.x1, bbox.y1)
        self.center = ((self.upper_left[0] + self.lower_right[0]) / 2,
                       (self.upper_left[1] + self.lower_right[1]) / 2)
        self.width = bbox.x1 - bbox.x0
        self.height = bbox.y1 - bbox.y0


    def __str__(self):
        return str(self.upper_left) + ",  " + str(self.lower_right)

    def add(self, rect):


        x0, y0 = rect.upper_left
        x1, y1 = rect.lower_right

        self.upper_left = (max(min(self.upper_left[0], x0),0),
                           max(min(self.upper_left[1], y0),0))
        self.lower_right = (max(self.lower_right[0], x1),
                           max(self.lower_right[1], y1))
        self.center = ((self.upper_left[0] + self.lower_right[0]) / 2,
                       (self.upper_left[1] + self.lower_right[1]) / 2)
        self.rect_num += 1
        self.width = self.lower_right[0] - self.upper_left[0]
        self.height = self.lower_right[1] - self.upper_left[1]



    def is_similar(self, rect, bound):

        x0, y0 = rect.upper_left
        x1, y1 = rect.lower_right
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        x_distance = abs(x - self.center[0])
        y_distance = abs(y - self.center[1])
        dist = sqrt(x_distance**2 + y_distance**2)
        w = (self.width +  x1 - x0) / 2
        h = (self.height + y1 - y0) / 2


        #little bit of hardcoding ( group more horizontaly than verticaly )
        return (y_distance < h * 0.8  and x_distance < w * 1.6 ) and dist < bound, dist




def plot_rectangles(img, rectangles):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.suptitle("preporucene_regije")
    ax.imshow(img)
    for rectangle in rectangles:
        rect = mpatches.Rectangle(rectangle.upper_left, # upper left
                              rectangle.lower_right[0] - rectangle.upper_left[0], #width
                              rectangle.lower_right[1] - rectangle.upper_left[1], #height
                                      fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

def logo_box(img, rectangles, show):

    '''
        Grouping rectangles (hierarchical grouping)

    '''

    sets = []
    for rectangle in rectangles:
        sets.append(RectangleSet(rectangle))

    while True:
        indices = (0, 0)
        min_distance = max(img.shape[0], img.shape[1])

        for i in range(len(sets) - 1):
            for j in range(i + 1, len(sets)):
                similar, distance =  sets[i].is_similar(sets[j], img.shape[1] * 0.5)
                if distance < min_distance and similar:
                    min_distance = distance
                    indices = (i, j)

        i, j = indices

        if indices == (0, 0) or len(sets) == 1: break
        sets[i].add(sets[j])
        del sets[j]

    #removing stupid boxes
    indices = []
    for i, s in enumerate(sets):
        height = s.height
        width = s.width
        area = height * width
        thresh = img.shape[0] * img.shape[1] / 8**2
        if area < thresh:
            indices.append(i)

    for i in reversed(indices):
        del sets[i]

    #plot
    if show:
        plot_rectangles(img, sets)



    #choosing logo (THE ONE) (3 options: max area, max mean or the highest logo)
    max_mean = 0
    index = 0
    max_area = 0
    min_x = img.shape[0]
    for i, s in enumerate(sets):
        mean = 0
        x0, y0 = s.upper_left
        x1, y1 = s.lower_right
        x0, x1, y0, y1 = np.int(x0), np.int(x1), np.int(y0), np.int(y1)
        mean = float(np.sum(img[y0:y1,x0:x1])) / ((x1 - x0) * (y1 - y0))
        area = (x1 - x0) * (y1 - y0)
        pos_x = (y1 + y0) / 2


        if pos_x < min_x:
            min_x = pos_x
            index = i

        # if mean > max_mean:
        #     max_mean = mean
        #     index = i


        # if area > max_area:
        #     max_area = area
        #     index = i

    # if no bboxes are found, return default
    if len(sets) == 0:
       return no_bbox(img)


    chosen = sets[index]


    rect = mpatches.Rectangle(chosen.upper_left, # upper left
                              chosen.lower_right[0] - chosen.upper_left[0], #width
                              chosen.lower_right[1] - chosen.upper_left[1], #height
                                      fill=False, edgecolor='red', linewidth=2)
    return rect



def no_bbox(img):
    h, w = img.shape
    rect = mpatches.Rectangle((np.int(0.1*w), np.int(0)), # upper left
                            np.int(0.8*w), #width
                            np.int(0.3*h), #height
                                    fill=False, edgecolor='red', linewidth=2)
    return rect


def remove_background2(img, stup):

    '''
        Hardcoded backgroung removing. Should be avoided as much as possible
        until we find something better
    '''

    i = 0
    height = img.shape[0]
    width = img.shape[1]
    height = img.shape[0]
    start_width = width
    start_height = height
    left_border = 0
    right_border = 0
    top_border = 0

    while(True):

        top = np.sum(img[0:stup, :])

        if top < 0.75 * stup * width: break

        img = img[stup:height, :]
        height = img.shape[0]
        top_border += stup
        if height == 0: break

    while(True):
        left = np.sum(img[:, 0:stup])
        if left < 0.25 * stup * height: break
        img = img[:, stup:width]
        left_border += stup
        width = img.shape[1]
        if width == 0: break

    while(True):
        right = np.sum(img[:, width-stup:width])

        if right < 0.25 * stup * height: break

        img = img[:, 0:width-stup]
        right_border += stup
        width = img.shape[1]
        if width == 0: break

    return img, left_border + 10, start_width - right_border - 10, top_border


def remove_top(img, stup):

    width = img.shape[1]
    height = img.shape[0]
    top_border = 0
    while(True):

        top = np.sum(img[0:stup, :])

        if top < 0.9 * stup * width: break

        img = img[stup-1:, :]
        height = img.shape[0]
        top_border += stup
        if height == 0: break

    return img[top_border:, :]


def add_safety_region(img, stup):
    '''
        Hardcoded safety
    '''

    img[:, 0:stup] = 0
    img[:, img.shape[1]-stup:img.shape[1]] = 1
    #img[0:stup,:] = 0
    img[img.shape[0]-2*stup:img.shape[1], : ] = 1
    return img

def remove_background(img_gray, show, filter_shape):


    filter_height, filter_width = filter_shape
    kernel = np.ones((filter_height, filter_width), np.float32)/ (filter_height * filter_width)

    img = cv2.filter2D(img_gray, -1, kernel)
    cv2.imshow('filtered', img)
    cv2.waitKey(0)

    edges = cv2.Canny(img, 100, 150, apertureSize = 3)
    cv2.imshow('canny', edges)
    cv2.waitKey(0)
    all_lines = cv2.HoughLines(edges,1,np.pi / (2*180), np.int(img.shape[0]/10))

    if all_lines is None:
        return img_gray

    min_x = img.shape[1]
    max_x = 0

    for line  in all_lines:
        for r, theta in line:
            if theta < 20*np.pi/180 or theta > 160*np.pi/180:
                a = np.cos(theta)

                # Stores the value of sin(theta) in b
                b = np.sin(theta)

                # x0 stores the value rcos(theta)
                x0 = a*r

                # y0 stores the value rsin(theta)
                y0 = b*r

                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000*(-b))

                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000*(a))

                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000*(-b))

                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000*(a))
                # if abs(x2-x1) > 50:
                #     continue


                x = np.int((x1 + x2) / 2)
                if x < min_x and x > 10:
                    min_x = x
                if x > max_x and x < img.shape[1]-10:
                    max_x = x
                cv2.line(img, (x1,y1), (x2,y2), (200,200,200), 5)

    if min_x > (img.shape[1] / 2):
        min_x = 0

    if max_x <  (img.shape[1] / 2) :
        max_x = img.shape[1]

    if show:
        plt.imshow(img)
        plt.show()

    del img, edges, all_lines
    return img_gray[:,min_x:max_x]




def find_logo(org_img, show=False):

    '''

        Image segmentation and logo extracting. First remove border using line detection.
        Then use only reciept and convert it again with new thresh. Afret all that,
        call grouping function (loxo_box) that finds logo

    '''

    img_gray = cv2.cvtColor( org_img, cv2.COLOR_RGB2GRAY )
    filter_width = 3
    filter_height = img_gray.shape[0] // 3

    img = remove_background(img_gray, show, (filter_height, filter_width))


    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    bw = img < thresh[0]
    bw = add_safety_region(bw, 10)

    # plt.imshow(bw)
    # plt.show()

    if show:
        fig, ax = plt.subplots(figsize=(5, 3))

    rectangles = set()

    label_image = label(bw)
    for region in regionprops(label_image):
        if region.area >= 70:
            # draw rectangles
            minr, minc, maxr, maxc = region.bbox
            width = maxr - minr + 10
            height = maxc - minc + 10
            rect = mpatches.Rectangle((minc - 5, minr - 5), height, width,
                                      fill=False, edgecolor='red', linewidth=2)
            area = height * width
            thresh_low = img.shape[0] * img.shape[1] / 15**2
            thresh_high = img.shape[0] * img.shape[1] / 3
            if area < thresh_low or area > thresh_high:
                continue
            if float(width) / height > 6 or float(height) / width > 6:
                continue
            rectangles.add(rect)

            if show:
                ax.add_patch(rect)

    if len(rectangles) == 0:
        print("Nema nista")
        logo_bbox = no_bbox(bw.copy())
    else:
        logo_bbox = logo_box(bw, rectangles, show)
    bbox = logo_bbox.get_bbox()



    logo = bw[np.int(bbox.y0):np.int(bbox.y1), np.int(bbox.x0):np.int(bbox.x1)]


    while logo.shape[0] < 60:

        logo = np.vstack((np.zeros(logo.shape[1])[np.newaxis, ...], logo))
        logo = np.vstack((logo, np.zeros(logo.shape[1])[np.newaxis, ...]))


        #print("Premali logo")


    while logo.shape[1] < 100:
        logo = np.hstack((np.zeros(logo.shape[0])[..., np.newaxis], logo))
        logo = np.hstack((logo, np.zeros(logo.shape[0])[..., np.newaxis]))

        #print("Premali logo")

    if show:

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        fig.suptitle("crno-bela s regijama")
        fig2.suptitle("odabrana regija")
        fig3.suptitle("logo")
        ax.imshow(bw)
        ax2.imshow(org_img)
        ax2.add_patch(logo_bbox)
        ax3.imshow(logo)


    # plt.imshow(logo)
    # plt.show()

    del img, img_gray, bw, label_image
    return logo
