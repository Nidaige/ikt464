import os
import numpy as numpy
import pandas
import torch
import pandas as pd


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



for item in listified_train_data:
   print(item[1])








