import os
import numpy as numpy
import pandas
#import torch
import pandas as pd
import struct
from pyTsetlinMachine.tm import *
import random
import math

'''
Function for float to binary from here: 
https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
'''
def binary(num):  # converts a float to binary, 8 bits
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def shuffle_dataset(dataset):
    output_values_list = []  # list for shuffled values
    output_labels_list = []  # list for shuffled labels
    output_raw_labels_list = []  # list for shuffled string labels
    tempdata = dataset  # copy dataset for further manipulation
    while len(tempdata[0])>0:  # as long as there is data in the dataset, keep going
        index = math.floor(random.random()*len(tempdata[0]))  # randomly select an element by index
        output_values_list.append(tempdata[0][index])  # copy the element in each slot (value, label, string label) to the output
        output_labels_list.append(tempdata[1][index])  # --||--
        output_raw_labels_list.append(tempdata[2][index])  # --||--
        for c in range(3):  # for each slot (value, label, string label):
            temp = tempdata[c][index]  # copy as a buffer variable
            tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
            tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
    return (np.array(output_values_list).astype(np.float),output_labels_list,output_raw_labels_list)
        
        
def iot_data_to_binary_list(path):
    data_values = []  # Holds numerical values of each parameter as list for each data entry
    data_labels = []  # Holds numerical representation of labels from entire dataset (0, 1, 2, 3, etc.)
    data_labels_raw = []  # Holds string representation of labels from entire dataset ('water_sensor', 'baby_monitor', 'motion_sensor', 'lights', etc)
    data_binary_values = []  # Holds binary representation of parameter list
    data = pandas.read_csv(path)  # read data csv
    for num in range(len(data)-1):  # numerically iterate through every line of data
        row = data.loc[num]  # get the data for each row
        datapoint = []  # empty list to hold the features of each row
        for item in row:  # for each value in a row
            datapoint.append(item)  # add it to the list of features for this row
        data_values.append(datapoint[0:-1])  # add the final list of features for this row to the processed dataset
        data_labels_raw.append(datapoint[-1])  # add string label to the list for that
        labels = list(set(data_labels_raw))  # get full list of label strings
        for i in data_labels_raw:  # for each string representation of a label
            data_labels.append(labels.index(i))  # translate to corresponding integer
    for item in data_values:  # for each dataset item
        rowie = ""  # string to temporarily hold binary representation of the data item
        for feature in item:  # for each value / feature in said item_
            rowie+=str(binary(float(feature)))  # concatenate the binary string for each feature to the string representing the item
        data_binary_values.append([*rowie])  # simultaneously splits the binary (bit stream representation) of the data into individual bits and adds it to the list of binary representations
    return [np.array(data_binary_values).astype(np.float), data_labels, data_labels_raw]  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

''' 
dataset from: https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification?resource=download
'''
#  path to data files
train_path = "data/iot_device_train.csv"
test_path = "data/iot_device_test.csv"

X_train, Y_train, labels_train = shuffle_dataset(iot_data_to_binary_list(train_path))
X_test, Y_test, labels_test = shuffle_dataset(iot_data_to_binary_list(test_path))


S = 9.0  # S-value
Clauses = 2500  # number of clauses to generate / to make each classification vote
T = 15  # T-value
print("Running "+str(Clauses)+", "+str(T)+", "+str(S))  # status report to the console
tm = MultiClassTsetlinMachine(Clauses, T, S)#, boost_true_positive_feedback=0)  # define the TM with above params
tm.fit(X_train, Y_train, epochs=50)  # train the TM for 50 epochs on training data
print("Training done, predicting...")
Prediction = tm.predict(X_test)
print("Predictions done, calculating score---")
Total = 0
Correct = 0
for test_data_sample in range(len(X_test)):
    Total += 1
    if Prediction[test_data_sample] == Y_test[test_data_sample]:
        Correct += 1
print("Accuracy: ", 100*Correct/Total , "%")