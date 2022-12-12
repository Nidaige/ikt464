import os
import numpy as np
import pandas
import pycuda
#import torch
import pandas as pd
#from pycuda import *
import pycuda
import struct
from PyTsetlinMachineCUDA.tm import *
import random
import math
import openpyxl

'''
Function for float to binary from here: 
https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
'''
def binary(num):  # converts a float to binary, 8 bits
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def shuffle_dataset(dataset):
    print("Shuffling...")
    output_values_list = []  # list for shuffled values
    output_labels_list = []  # list for shuffled labels
    tempdata = dataset  # copy dataset for further manipulation
    while len(tempdata[0])>0:  # as long as there is data in the dataset, keep going
        index = math.floor(random.random()*len(tempdata[0]))  # randomly select an element by index
        output_values_list.append(np.array(tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
        output_labels_list.append(tempdata[1][index])  # --||--
        for c in range(2):  # for each slot (value, label, string label):
            #temp = tempdata[c][index]  # copy as a buffer variable
            tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
            tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
    print("Shuffling done")
    return (output_values_list, output_labels_list)

def iot_data_to_binary_list(path, max_bits, database, registry):
    print("Binarizing...")
    data_values = []
    data = pandas.read_csv(path)  # read data csv
    # Duplicate the registry, add new keys to it
    reg = registry
    db = database
    for new_label in list(data[' Label'].unique()):
        if new_label not in reg.keys():
            reg[new_label] = 0
            db [new_label] = []
    
    for num in range(len(data)-1):  # numerically iterate through every line of data
        row = data.loc[num]  # get the data for each row
        datapoint = []  # empty list to hold the features of each row
        for item in row:  # for each value in a row
            datapoint.append(item)  # add it to the list of features for this row
        data_values.append(datapoint)  # add the final list of features for this row to the processed dataset  

    for item in data_values:  # for each dataset item
        values = item[0:-1]
        label = item[-1]
        rowie = ""  # string to temporarily hold binary representation of the data item
        for feature in values:  # for each value / feature in said item_
            rowie+=str(binary(float(feature)))[-max_bits:]  # concatenate the binary string for each feature to the string representing the item
        db[label].append([*rowie])
        registry[label] += 1
    print("Binarizing done")
    return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

''' 
dataset from: https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification?resource=download
'''
#  path to data files
#data_paths = ["data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "data/Wednesday-workingHours.pcap_ISCX.csv", "data/Monday-WorkingHours.pcap_ISCX.csv"]
data_paths = ["data/Wednesday-workingHours.pcap_ISCX.csv", "data/Monday-WorkingHours.pcap_ISCX.csv"]
maximum_bits = 16
max_data = 100000
minimum_data_in_category = 5000
all_data_dict = {}
class_registry = {}
all_data = [[],[]]
smallest_count = 0
confusion_matrix = {}
# Read files and create data
for path in data_paths:
    print("Pre-processing data from", path)
    all_data_dict, class_registry = iot_data_to_binary_list(path, maximum_bits, all_data_dict, class_registry)
counts = []
for key in class_registry.keys():
    number_of_that_class = class_registry[key]  # get registry data for <key> category
    if number_of_that_class > minimum_data_in_category:  # if number of entries for <key> category is above threshold, use it in list of counts
        counts.append(number_of_that_class)
smallest_count = min(counts)  # get smallest count above threshold
for n in range(min(math.floor(max_data/len(class_registry.keys())),smallest_count)):  # from 0 to n where n is the lesser of max data number/# of categories OR n is == the smallest count above the threshold
    for key in all_data_dict.keys():
        number_of_that_class = class_registry[key]
        if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
            all_data[0].append(all_data_dict[key][n])
            all_data[1].append(key)
            confusion_matrix[key] = {}
# initialize confusion matrix
for key in confusion_matrix.keys():
    for key2 in confusion_matrix.keys():
        confusion_matrix[key][key2] = 0

print("data_distribution:")
print(class_registry)
X_all_data, Y_all_data = shuffle_dataset(all_data)  # X_all_data: list of lists of features | y_all_data: 0/1/2/... , but wrong. Fixing here. | labels_all_data: string of the class of each object
print("Converting String labels to numbers...")
labels_all_data_set = list(set(Y_all_data))  # create set of ALL string-labels
for i in range(len(Y_all_data)):  # for each element
    Y_all_data[i] = labels_all_data_set.index(Y_all_data[i])  # assign the true index to each data label Y


print("Done converting labels.")
print("Converting to numpy arrays...")
X_all_data = np.array(X_all_data).astype(float)
Y_all_data = np.array(Y_all_data)
print("Done converting to numpy arrays")
count = len(X_all_data)
split = 0.7
print("Splitting into training/test with a ", 100*split,"% split...")
X_train = X_all_data[0:math.floor(count*split)]
Y_train = Y_all_data[0:math.floor(count*split)]

X_test = X_all_data[math.floor(count*split):]
Y_test = Y_all_data[math.floor(count*split):]
print("Done splitting")
print("Initializing variables and starting TM...")
S = [10]  # S-value
Clauses = 25000  # number of clauses to generate / to make each classification vote
T = [30]  # T-value
Epochs = 1000
Batch_size = 100
print("# of labels: ",len(labels_all_data_set), labels_all_data_set)
for s_ in S:
    for t_ in T:
        print("Running clauses:"+str(Clauses)+", T:"+str(t_)+", S:"+str(s_))  # status report to the console
        tm = MultiClassTsetlinMachine(Clauses, t_, s_, boost_true_positive_feedback=0)  # define the TM with above params
        tm.fit(X_train, Y_train, epochs = Epochs, batch_size = Batch_size)  # train the TM for 50 epochs on training data
        print("Training done, predicting...")
        Prediction = tm.predict(X_test)
        print("Predictions done, calculating score---")
        Total = 0
        Correct = 0
        conf_matr = confusion_matrix
        for test_data_sample in range(len(X_test)):
            Total += 1
            if Prediction[test_data_sample] == Y_test[test_data_sample]:  # if correct guess:
                Correct += 1
            conf_matr[labels_all_data_set[Prediction[test_data_sample]]][labels_all_data_set[Y_test[test_data_sample]]] += 1
        for key in conf_matr.keys():
            for key2 in conf_matr.keys():
                confusion_matrix[key][key2] = confusion_matrix[key][key2]/Total
        dict1 = confusion_matrix
        df = pd.DataFrame(data=dict1, index=list(confusion_matrix.keys()))
        df.to_excel("uberbalance results/new_best/data.xlsx")
        print("count",count)        
        print("Accuracy: ", 100*Correct/Total , "%")
        path = "uberbalance results/new_best/S-"+str(s_)+"_T-"+str(t_)+"_Clauses-"+str(Clauses)+"_Epochs-"+str(Epochs)+"_BatchSize-"+str(Batch_size)+".txt"
        file = open(path,"w")
        file.write("Results: \n")
        file.write("Dataset: Monday and Wednesday.\n")
        file.write("Params: S: "+str(s_)+ ", T:"+str(t_)+", Clauses:"+str(Clauses)+", Epochs: "+str(Epochs)+"\n")
        file.write("Accuracy: "+str(100*Correct/Total)+"% \n")
        file.close()
