import cv2
import loadImages
import sys
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from layers.layers import spatial_pooling_forward
import regions

def main():


    for i in range(30):
        img_dir = os.getcwd() + "/test/test"
        img = cv2.imread(img_dir + "/" + str(i) + ".jpg")

        logo = regions.find_logo(img[:np.int(img.shape[0]/2.5), :], show=False)

        logo = spatial_pooling_forward(logo, (60, 100))
        plt.imshow(logo)
        plt.show()

    print(img_dir + "0.jpg")
    plt.imshow(img)
    plt.show()








if __name__ == "__main__":
    main()
