from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from regions import find_logo
from loadImages import load_images
from math import ceil
from layers.layers import spatial_pooling_forward
import pickle
import atexit
import os

'''
This file is going through train file and creates logos from batches
of pictures. Batches are saved in "batches" folder in .pkl format and
in train_logos folder as .png.

'''

#TODO load image dimensions from params.pkl



def main():
    current_dir = os.getcwd()
    train_dir = current_dir + "/train"
    logo_dir = current_dir + "/train_logos"
    batch_dir = current_dir + "/batches"
    batch_size = 1000

    start = time.time()
    # file storing start_from, num_batches and num_images variables

    global start_from
    global num_batches
    global num_images

   
    f = open('params.pkl', 'rb')
  
    # start_from = num_batches = num_images = 0
    
    # pickle.dump(start_from, f)
    # pickle.dump(num_batches, f)
    # pickle.dump(num_images, f)

    start_from = pickle.load(f)
    num_batches = pickle.load(f)
    num_images = pickle.load(f)
    f.close()

    

    #create 4 new batches
    for i in range(30):
        num_batches += 1
        f = open(current_dir + "/batches/batch" + str(num_batches) + ".pkl", 'wb')
        train, labels = load_images(train_dir, batch_size, start_from)

        #inputs and labels for stacking
        X = np.empty((1, 60, 100))
        y = np.empty((1, 25))

        for k, img in enumerate(train):
            h, w, _ = img.shape
            img = img[:np.int(h/2.5), np.int(0.05*w):np.int(0.95*h), :]
            logo = find_logo(img, show=False)

            logo = spatial_pooling_forward(logo, (60, 100))
            #saving picture of logo in train_logos
            cv2.imwrite(logo_dir + "/" + str(num_images) + ".jpg", logo*255 )
            num_images += 1

            X = np.concatenate((X, logo[np.newaxis, ...]))
            y = np.concatenate((y, labels[k][np.newaxis, ...]))
        #saving logos in batches folder
        X = X[1:, ...]
        y = y[1:, ...]

        pickle.dump(X, f)
        pickle.dump(y, f)
        start_from += np.int(ceil(batch_size / 25))
        print("Generated " + str( (i+1) * batch_size) + " logos")
        end = time.time()
        print("Preslo je: " + str(end-start) + " sekundi!")


if __name__ == "__main__":
    main()



#saving current variables on exit
def on_exit():
    global start_from
    global num_batches
    global num_images
    f = open('params.pkl', 'wb')
    pickle.dump(start_from, f)
    pickle.dump(num_batches, f)
    pickle.dump(num_images, f)
    f.close()

atexit.register(on_exit)
