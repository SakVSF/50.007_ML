#part1
import numpy as np
import random
from collections import Counter
#define datapaths 
en_train_path = "EN/train"
en_dev_in_path = "EN/dev.in"
en_dev_out_path ="EN/dev.out"
en_dev_p1_out_path = "EN/dev.p1.out"


fr_train_path = "FR/train"
fr_dev_in_path = "FR/dev.in"
fr_dev_out_path ="FR/dev.out"
fr_dev_p1_out_path = "FR/dev.p1.out"

N = 18 #no of unique labels 
#add START and END to the labels 
labels = {"START": 0,
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
          #"I-PRT":15,
          "B-SBAR":15,
          "I-SBAR":16,
          "B-VP": 17,
          "I-VP":18,
          "END": 19}
labels_list = ["START","O", "B-ADJP", "I-ADJP","B-ADVP","I-ADVP","B-CONJP","I-CONJP","B-INTJ","I-INTJ","B-NP","I-NP", "B-PP","I-PP","B-PRT", "B-SBAR","I-SBAR","B-VP","I-VP","END"]

# Read training data
def read_training_data(path):
    results = []
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if len(line.strip().rsplit(" ", 1)) == 2:
                token, label = line.strip().rsplit(" ", 1)
                #print(label)
              
                results.append((token, labels[label]))
            else:
                continue
    return results

    '''
    results = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            token, label = line.rsplit(" ", 1)
            if label in labels:
                results.append((token, labels[label]))
    return results'''
    



# Read dev.in data
def read_dev_in_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


# Read dev.out data
def read_dev_out_data(path):
    results = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token, label = line.rsplit(' ', 1)
                results.append((token, labels[label]))
    return results

def calculate_number_of_labels(data):
    '''label_counts = {}
    for elem in data:
        label = elem[1]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    print("label_counts:", label_counts)
    return label_counts
    '''
    return Counter(elem[1] for elem in data)


def get_all_tokens(data):
    '''
    unique_tokens = []
    for item in data:
        if item[0] not in unique_tokens:
            unique_tokens.append(item[0])
    return unique_tokens
    '''
    return list(set(item[0] for item in data))


def calculate_emission_parameters(data, all_tokens, k=1.0):
    print("all tokens", len(all_tokens), "N", N)
    # the extra +1 column is for #UNK# tokens
    emission_counts = np.zeros((N, len(all_tokens) + 1), dtype=np.longdouble)   #creates an NP array where rows - labels, columns - unique tokens
    emission_parameters = np.zeros((N, len(all_tokens) + 1), dtype=np.longdouble)

    # calculate count(label -> token) 
    for token, label in data:
        emission_counts[label - 1][all_tokens.index(token)] += 1    #keep a count of how many times each token appears for each label in the data list
    #last column filled with k value for UNK token
    emission_counts[:, -1] = [k] * N
    '''
    # calculate count(label)
    label_counts = []
    label_counts_dict = calculate_number_of_labels(data)    # a dictionary with the format : {label:count} , i.e, each key is a label which has its count as the corresponding value
   
    sorted_labels = sorted(label_counts_dict.keys())
    print("sorted labels", sorted_labels)
    for label in sorted_labels:
        print("adding", label, ":",label_counts_dict[label] )
        label_counts.append(label_counts_dict[label])
    label_counts = np.array(label_counts)                # in increasing order of label (from 0 to 8), contains (label:label count)
    print("label_counts",label_counts)

'''

    label_counts = np.array(list(val[1] for val in sorted(calculate_number_of_labels(data).items())))
    print("label_counts",label_counts)
    # calculate count(label->token)/count(label) for each label 
    for index, value in enumerate(emission_counts): 
        emission_parameters[index] = emission_counts[index] / (label_counts[index] + k)  #index - each row of 2d array


    return emission_parameters  #each value in this array gives the probability e(x|y) -> token given a label


# Get tag from word
def get_label_from_token(token, emission_parameters, all_tokens):
    if token in all_tokens:
        column_to_consider = emission_parameters[:, all_tokens.index(token)]   
    else:
        column_to_consider = emission_parameters[:, -1]
        
    # Get the indices of maximum values in the array
    max_indices = np.where(column_to_consider == column_to_consider.max())[0]
    # Randomly choose an index from the maximum indices
    x = random.choice(max_indices) + 1
    return labels_list[x]


def write_prediction_output_to_file(language):
    if language == "EN":
        # Conduct training/supervised learning (M-Step)
        train_data = read_training_data(en_train_path)
        all_tokens = get_all_tokens(train_data)
        emission_parameters = calculate_emission_parameters(train_data, all_tokens)

        # Execute testing/decoding (E-Step)
        predicted_results = []
        test_data = read_dev_in_data(en_dev_in_path)
        for token in test_data:
            if token:
                predicted_results.append(token + " " + get_label_from_token(token, emission_parameters, all_tokens ))
            else:
                predicted_results.append("")
        with open(en_dev_p1_out_path, "w+", encoding="utf-8") as file:
            for line in predicted_results:
                file.write(line + "\n")

    elif language == "FR":
        # Conduct training/supervised learning (M-Step)
        train_data = read_training_data(fr_train_path)
        all_tokens = get_all_tokens(train_data)
        emission_parameters = calculate_emission_parameters(train_data, all_tokens)

        # Execute testing/decoding (E-Step)
        predicted_results = []
        test_data = read_dev_in_data(fr_dev_in_path)
        for token in test_data:
            if token:
                predicted_results.append(token + " " + get_label_from_token(token, emission_parameters, all_tokens))
            else:
                predicted_results.append("")
        with open(en_dev_p1_out_path, "w+", encoding="utf-8") as file:
            for line in predicted_results:
                file.write(line + "\n")
for language in ["EN", "FR"]:
    write_prediction_output_to_file(language)