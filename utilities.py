import numpy as np
import random

'''
This file serves as a collection of all such functions which will be repeatedly used in all the parts (part1.py, part2.py, part3.py) files
This file is then imported into the other files so that the following common functions can be accessed
'''

#define datapaths 
en_train_path = "EN/train"
en_dev_in_path = "EN/dev.in"
en_dev_out_path ="EN/dev.out"
en_dev_p1_out_path = "EN/dev.p1.out"
en_dev_p2_out_path = "EN/dev.p2.out"

fr_train_path = "FR/train"
fr_dev_in_path = "FR/dev.in"
fr_dev_out_path ="FR/dev.out"
fr_dev_p1_out_path = "FR/dev.p1.out"
fr_dev_p2_out_path = "FR/dev.p2.out"


N_FR = 7 #no of unique labels in FR train dataset 
#add START and STOP to the labels 
labels_FR = {"START":0,
          "O": 1,
          "B-positive":2,
          "I-positive":3,
          "B-neutral": 4,
          "I-neutral": 5,
          "B-negative":6,
          "I-negative": 7,
          "STOP": 8}
labels_list_FR = ["START", "O", "B-positive", "I-positive", "B-neutral", "I-neutral", "B-negative", "I-negative", "STOP"]

N_EN = 18 #no of unique labels in EN train dataset 
#add START and STOP to labels 
labels_EN = {"START": 0,
          "O": 1,
          "B-ADJP":2,
          "I-ADJP":3,
          "B-ADVP":4,
          "I-ADVP":5,
          "B-CONJP":6,
          "I-CONJP":7,
          "B-INTJ": 8,
          "I-INTJ": 9,
          "B-NP": 10,
          "I-NP": 11,
          "B-PP": 12,
          "I-PP": 13,
          "B-PRT":14,
          "B-SBAR":15,
          "I-SBAR":16,
          "B-VP": 17,
          "I-VP":18,
          "STOP": 19}
labels_list_EN = ["START","O", "B-ADJP", "I-ADJP","B-ADVP","I-ADVP","B-CONJP","I-CONJP","B-INTJ","I-INTJ","B-NP","I-NP", "B-PP","I-PP","B-PRT", "B-SBAR","I-SBAR","B-VP","I-VP","STOP"]

# Read training data
def read_training_data(path, labels):
    
    results = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            token, label = line.rsplit(" ", 1)
            if label in labels:
                results.append((token, labels[label]))
    return results

    

# Read dev.in data
def read_dev_in_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


'''# Read dev.out data
def read_dev_out_data(path, labels):
    results = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token, label = line.rsplit(' ', 1)
                results.append((token, labels[label]))
    return results

'''
# Calculate the number of times each label/tag occurs and store it in a dictionary {label:label count}
def get_label_counts(data):
    label_counts_dict = {}
    for elem in data:
        label = elem[1]
        if label in label_counts_dict:
            label_counts_dict[label] += 1
        else:
            label_counts_dict[label] = 1
    print("label_counts_dictionary:", label_counts_dict)
  
    
    label_counts = []
       # a dictionary with the format : {label:count} , i.e, each key is a label which has its count as the corresponding value
   
    sorted_labels = sorted(label_counts_dict.keys())
    #print("sorted labels", sorted_labels)
    for label in sorted_labels:
        #print("adding", label, ":",label_counts_dict[label] )
        label_counts.append(label_counts_dict[label])

    label_counts = np.array(label_counts)  
    print("label counts:", label_counts)              # in increasing order of label (from 0 to 8), contains label counts
    
    return label_counts

   

# Returns a list of all unique tokens/words in the dataset 
def get_tokens(data):
    unique_tokens = []
    for item in data:
        if item[0] not in unique_tokens:
            unique_tokens.append(item[0])
    return unique_tokens
  

# Predict label/tag from token/word using the emission probabilities table 
def label_from_token(token, emission_parameters, all_tokens, labels_list):
    if token in all_tokens:
        column_to_consider = emission_parameters[:, all_tokens.index(token)]   #column correspinding to the token in emission_parameters 2d array
    else:
        column_to_consider = emission_parameters[:, -1]   #if it is a new word that appears in test set but not training set, assign 1 to it
        
    # Get the indices of maximum values in the array
    max_indices = np.where(column_to_consider == column_to_consider.max())[0]   #find the maximum probabilities 

    # Since there can be multiple maximum probabilities, randomly choosing an index from the maximum indices
    x = random.choice(max_indices) + 1
    return labels_list[x]
