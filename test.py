from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from regions import find_logo
from loadImages import load_images
from math import ceil
from layers.layers import spatial_pooling_forward
from solver import Solver
from cnn import ThreeLayerConvNet
import pickle
import atexit
import time
import sys
import os
import csv
import sys

''' 
    TODO again, picture dimensions loading over params in params.pkl
    Main file which classifies logos. 

'''



def main():

    mode = str(sys.argv[1])
    if mode != "train" and mode != "test":
        print( "Enter mode: train or test!")
        sys.exit(0)
    current_dir = os.getcwd()
    batch_dir = current_dir + "/batches"
    test_dir = current_dir + "/test"

    # one fifth of all logos for validation, rest for training
    train_part = 1
    val_part = 0.2 * train_part


    classes = ["Albertsons", "BJs", "CVSPharmacy", "Costco", "FredMeyer", "Frys",
                "HEB", "HarrisTeeter", "HyVee", "JewelOsco", "KingSoopers", "Kroger", "Meijer", "Publix",
                "Safeway", "SamsClub", "ShopRite", "Smiths", "StopShop", "Target", "Walgreens",
                "Walmart", "Wegmans", "WholeFoodsMarket", "WinCoFoods", "Other"]

    


    model = ThreeLayerConvNet(input_dim=(1, 60, 100),
                            filter_size=7, hidden_dim=1000,
                            num_classes=25, weight_scale=5e-2, reg = 5e-6)


    if mode == "train":
        # loading train data

        f = open(batch_dir + '/batch' + str(1) + '.pkl', 'rb')
        X = pickle.load(f)
        y = pickle.load(f)
        f.close()
                  
        for i in range(1,4):
            f = open(batch_dir + '/batch' + str(i+1) + '.pkl', 'rb')
            xi = pickle.load(f)
            yi = pickle.load(f)

            # mean = np.mean(xi,axis=(1,2))
            #
            # for j, m in enumerate(mean):
            #         xi[j, ...] -= m

            X = np.concatenate((X, xi))
            y = np.concatenate((y, yi))

            f.close()


        # permutating and creating train and validation sets
        permutated_indices_train = np.random.permutation(X.shape[0])
        X = X[:, np.newaxis, ...]
        if np.isnan(np.sum(X)):
            print("NaN")
        X = X[permutated_indices_train, ...]
        y = y[permutated_indices_train, ...]

        val_size = np.int(X.shape[0] * val_part)
        train_size = np.int(X.shape[0] * (train_part - val_part))

        X_train = X[:train_size]
        X_val = X[train_size:]

        y_train = y[:train_size]
        y_val = y[train_size:]


        data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
        }
        if np.isnan(np.sum(X)):
            print("NaN")

        solver = Solver(model, data,
                    num_epochs=15, batch_size=500,
                    update_rule='adam',
                    optim_config={
                    'learning_rate': 1e-2,
                    'lr_decay': '0.92'
                    },
                    verbose=True, print_every=1,
                    checkpoint_name="net_params")

        solver.train()

    if mode == "test":

        model.load_params("net_params_epoch_6.pkl")
        test = load_images(test_dir)

        f = open(current_dir +  'labels.csv', 'wb')
        X = np.zeros((60, 100))
        X = X[np.newaxis, ...]
        for i in range(len(test)):
            logo = find_logo(test[i][:np.int(test[i].shape[0]/2.5), :], show=False)

            logo = spatial_pooling_forward(logo, (60, 100))     
            plt.imshow(logo)
            plt.show()
            X = np.concatenate((X, logo[np.newaxis, ...]))

        X = X[1:, ...]

        scores = model.loss(X[:, np.newaxis, ...])


        y_pred = []

        for i in range(scores.shape[0]):
            #print np.int(np.argmax(scores[i, ...]))
            y_pred.append(classes[np.int(np.argmax(scores[i, ...]))])
            print classes[np.int(np.argmax(scores[i, ...]))]
            pickle.dump(classes[np.int(np.argmax(scores[i, ...]))] + "\n", f)

        f.close()


if __name__ == "__main__":
    main()
