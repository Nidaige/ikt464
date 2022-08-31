import os
import numpy as numpy
import pandas
import torch
import pandas as pd

def getlongest(a,b):
   if len(a)<len(b):
      return b
   else:
      return a
''' 
dataset from: https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification?resource=download
'''

train_path = "data/iot_device_train.csv"
test_path = "data/iot_device_test.csv"


load_train = pandas.read_csv(train_path)
for num in range(len(load_train)):
   row = load_train.loc[num]
   print(row)
   print("-------------------------------------------------------------")





