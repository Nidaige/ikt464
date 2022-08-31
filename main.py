import os
import numpy as numpy
import pandas
import torch
import pandas as pd
import struct

'''
Function for float to binary from here: 
https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
'''
def binary(num):  # converts a float to binary
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

''' 
dataset from: https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification?resource=download
'''
#  path to data files
train_path = "data/iot_device_train.csv"
test_path = "data/iot_device_test.csv"

# empty lists to hold the data once loaded
listified_train_data = []
listified_test_data = []


# code for loading data from csv into lists
load_train = pandas.read_csv(train_path)  # read training data csv
for num in range(len(load_train)):  # numerically iterate through every line of data
   row = load_train.loc[num]  # get the data for each row
   datapoint = []  # empty list to hold the features of each row
   for item in row:  # for each value in a row
      datapoint.append(item)  # add it to the list of features for this row
   listified_train_data.append([datapoint[0:-1],datapoint[-1]])  # add the final list of features for this row to the processed dataset

binary_train_data = []

for item in listified_train_data:
    #print(item)
    rowie = ""

    for feature in item[0]:
        rowie+=str(binary(float(feature)))
    binary_train_data.append([*rowie])  # simultaneously splits the binary (bit stream representation) of the data into individual bits and adds it to the list of binary representations

for i in binary_train_data:
   print(i)










