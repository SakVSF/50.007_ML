#part1
from utilities import *
import numpy as np
import random
from collections import Counter

def estimate_emission_parameters(data, all_tokens, N, k=1.0):

    print("number of unique tokens", len(all_tokens), "N", N)

    # the extra +1 column is for #UNK# tokens
    emission_counts = np.zeros((N, len(all_tokens) + 1), dtype=np.longdouble)   #creates an NP array where rows - labels, columns - unique tokens
    emission_parameters = np.zeros((N, len(all_tokens) + 1), dtype=np.longdouble)

    # calculate count(label -> token) 
    for token, label in data:
        emission_counts[label-1][all_tokens.index(token)] += 1    #keep a count of how many times each token appears for each label in the data list
    
    #last column filled with k value for UNK token
    emission_counts[:, -1] = [k] *(N)
   
    # calculate count(label)
    label_counts = get_label_counts(data)    # a list where label_counts[0] gives the count of label 0 which is START
   
    # calculate count(label->token)/count(label) for each label 
    for index in range(len(emission_counts)):
        emission_parameters[index, :] = emission_counts[index, :] / (label_counts[index] + k)

    print("Emission table:", emission_parameters[:2][:2])


    return emission_parameters  #each value in this 2d array gives the probability e(x|y) with corresponding row as the label/tag and column as the token/word





   
        



def predict_FR_output():
    train_data = read_training_data(fr_train_path, labels_FR)
    all_tokens = get_tokens(train_data)
    emission_parameters = estimate_emission_parameters(train_data, all_tokens, N_FR)

    predictions = []
    test_data = read_dev_in_data(fr_dev_in_path)
    for token in test_data:
        if token:
            label = label_from_token(token, emission_parameters, all_tokens, labels_list_FR)
            predictions.append(token + " " + label )
        else:
            predictions.append("")

    with open(fr_dev_p1_out_path, "w+", encoding="utf-8") as file:
        for line in predictions:
            file.write(line + "\n")

def predict_EN_output():
    train_data = read_training_data(en_train_path, labels_EN)
    all_tokens = get_tokens(train_data)
    emission_parameters = estimate_emission_parameters(train_data, all_tokens, N_EN)

    predictions = []
    test_data = read_dev_in_data(en_dev_in_path)
    for token in test_data:
        if token:
            label = label_from_token(token, emission_parameters, all_tokens, labels_list_EN)
            predictions.append(token + " " +  label )
        else:
            predictions.append("")

    with open(en_dev_p1_out_path, "w+", encoding="utf-8") as file:
        for line in predictions:
            file.write(line + "\n")



predict_EN_output()
predict_FR_output()