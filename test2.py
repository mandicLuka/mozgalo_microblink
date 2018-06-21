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
from cnn2 import ThreeLayerConvNet
import pickle
import atexit
import time
import sys
import os
import csv
import sys
import gc



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
                            filter_sizes=(7, 3), hidden_dims=(1000, 1000),
                            num_classes=25, weight_scale=5e-2, dropout=0.5,
                            num_filters=50)


    if mode == "train":
        # loading train data

        model.load_params("net_params_11_epoch_2.pkl")
        
        batch_names = os.listdir(batch_dir)
        num_batches = 34 #int(batch_names[-1][-5:-4])
        X_val = np.zeros((1, 1, 60, 100))
        y_val = np.zeros((1, 25))
        learning_rate = 1e-4  #default is 1e-3
        lr_decay = 0.98
        print (num_batches)

        



        for i in range(11, num_batches+1):
            f = open(batch_dir + '/batch' + str(i) + '.pkl', 'rb')
            X = pickle.load(f)
            y = pickle.load(f)  
            f.close()


            # permutating and creating train and validation sets
            permutated_indices_train = np.random.permutation(X.shape[0])
            X = X[:, np.newaxis, ...]

            if np.isnan(np.sum(X)):
                print("NaN")
            
            X = X[permutated_indices_train, ...]
            y = y[permutated_indices_train, ...]

            train_size = np.int(X.shape[0] * (train_part - val_part))

            X_train = X[:train_size, ...]

            X_val = np.concatenate((X_val, X[train_size:, ...]))

            y_train = y[:train_size]
            y_val = np.concatenate((y_val, y[train_size:, ...]))



            if y_val.shape[0] > 499:
                val_size = np.int(y.shape[0] * val_part)
                X_val = X_val[val_size:]
                y_val = y_val[val_size:]

            if np.sum(y_val[0, :]) == 0:
                X_val = X_val[1:, ...]
                y_val = y_val[1:, ...]

            data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
            }

            try:
                f = open("net_params_" + str(i-1) + "_epoch_1.pkl")
                model.load_params("net_params_" + str(i-1) + "_epoch_1.pkl")
                f.close()
            except (IOError, EOFError) as e:
                print("No params")

            solver = Solver(model, data,
                        num_epochs=2, batch_size=100,
                        update_rule='adam',
                        optim_config={
                        'learning_rate': learning_rate,
                        'lr_decay': lr_decay
                        },
                        verbose=True, print_every=1,
                        checkpoint_name="net_params_" + str(i))

            
            solver.train()

            for k in solver.optim_configs:
                learning_rate = solver.optim_configs[k]['learning_rate'] 

            del X_train, y_train, data

    if mode == "test":

        model.load_params("net_params_12_epoch_2.pkl")
        f = open('labels.csv', 'wb')
        writer = csv.writer(f)
        
        for i in range(100):
            test = load_images(test_dir, startFrom = i*100)
            print("Done with " + str((i+1)*100))
            
            X = np.zeros((1,60,100))
           
            for i in range(len(test)):
                logo = find_logo(test[i][:np.int(test[i].shape[0]/2.5), :], show=False)

                logo = spatial_pooling_forward(logo, (60, 100))     
                plt.imshow(logo)
                plt.show()
                X = np.concatenate((X, logo[np.newaxis, ...]))

            X = X[1:, ...]

            scores = model.loss(X[:, np.newaxis, ...])

            shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
            Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

            log_probs = shifted_logits - np.log(Z)
            probs = np.exp(log_probs)

            for i in range(probs.shape[0]):
                #print probs[i]
                #print np.int(np.argmax(scores[i, ...]))
                
                pred_class = np.max(probs[i])
                if pred_class > 0.46:
                    writer.writerow([classes[np.int(np.argmax(probs[i]))]])
                    #print classes[np.int(np.argmax(probs[i]))]
                else:
                    writer.writerow(["Other"])
                    #print "Other"
                
            del X, test
        f.close()

if __name__ == "__main__":
    main()
