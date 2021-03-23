# path tools:
import sys,os
sys.path.append(os.path.join(".."))
import argparse

#Natural networks with numpy
import numpy as np
from utils.neuralnetwork import NeuralNetwork


#Machine learning tools:
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import fetch_openml

import argparse



def main():
    
    '''
    --------------Defining command line arguments---------------
    '''
    ap = argparse.ArgumentParser(description = "[INFO] creating benchmark classifier")
    
    #1. argument: Number of hidden layers in first pile of hidden layers:
    ap.add_argument("-hl1", #flag
                    "--hidden_layer_1", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=32, # If you don't the default is 32 hiddenlayers.
                    type=int, # The input has has to be a integer.
                    help="The the number of hidden layers.")
    
    #2. argument: Number of hidden layers in second pile of hidden layers:
    ap.add_argument("-hl2", #flag
                    "--hidden_layer_2", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=0,
                    type=int, # The input has has to be a integer.
                    help="The the number of hidden layers.")
    
    #3. argument: Number of hidden layers in third pile of hidden layers:
    ap.add_argument("-hl3", #flag
                    "--hidden_layer_3", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=0,
                    type=int, # The input has has to be a integer.
                    help="The the number of hidden layers.")
    
    #4. argument: number of epochs the data should train on:
    ap.add_argument("-epochs",
                    "--number_of_epochs",
                    required=False,
                    default=100,
                    type=int,
                    help="The number of times the data is run through")
    
    args = vars(ap.parse_args())
    
    
    # Putting the arguments into variables:
    hidden_layer_1 = args["hidden_layer_1"]
    hidden_layer_2 = args["hidden_layer_2"]
    hidden_layer_3 = args["hidden_layer_3"]
    number_of_epochs = args["number_of_epochs"]
    
        
    '''
    -----------------------Downloading and cleaning data---------------------------
    '''
    # Fetching/downloading data set
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    #X is the the images, y is the the category.
    X = np.array(X)
    y = np.array(y)
    
    X = (X - X.min())/(X.max() - X.min())
    
    
    #Creating training and test data.
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        #random_state=9,
                                                        train_size=7500, 
                                                        test_size=2500)
    
    # convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    
    
    '''
    --------------------Creating different options for piles of hidden layers--------------------
    '''

    #train network:
    
    # If only 1 argument of hidden layers are givin:
    if hidden_layer_1 > 0 and hidden_layer_2 == 0 and hidden_layer_3 == 0:
        
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], hidden_layer_1, 10]) #CLI-argument
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=number_of_epochs) #CLI-argument
        print("___________ 2 args _____________") #For my self so i can see that the code took 2 arguments
        
        
        # evaluate network
        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))
        
        
    # If 2 argument of hidden layers are givin:
    elif hidden_layer_1 > 0 and hidden_layer_2 > 0 and hidden_layer_3 == 0:
        
        
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], hidden_layer_1, hidden_layer_2, 10]) #CLI-argument
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=number_of_epochs) #CLI-argument
        print("___________ 3 args _____________") #For my self so i can see that the code took 3 arguments
        
        # evaluate network
        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))
        
        
        
    # If 3 argument of hidden layers are givin:    
    elif hidden_layer_1 > 0 and hidden_layer_2 > 0 and hidden_layer_3 > 0:
        
        
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], hidden_layer_1, hidden_layer_2, hidden_layer_3, 10]) #CLI-argument
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=number_of_epochs) #CLI-argument
        print("___________ 4 args _____________") #For my self so i can see that the code took 4 arguments
        
        # evaluate network
        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))


if __name__ =='__main__':
    main()
    