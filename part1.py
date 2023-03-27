#part1
import numpy as np
 

#define datapaths 
en_train_path = "EN/train"
en_dev_in_path = "EN/dev.in"
en_dev_out_path ="EN/dev.out"
en_dev_p1_out_path = "EN/dev.p1.out"


fr_train_path = "FR/train"
fr_dev_in_path = "FR/dev.in"
fr_dev_out_path ="FR/dev.out"
fr_dev_p1_out_path = "FR/dev.p1.out"

N = 7  #no of unique labels 
#add START and END to the labels 
labels = {"START": 0,
          "O": 1,
          "B-positive": 2,
          "I-positive": 3,
          "B-neutral": 4,
          "I-neutral": 5,
          "B-negative": 6,
          "I-negative": 7,
          "END": 8}
labels_list = ["START", "O", "B-positive", "I-positive", "B-neutral", "I-neutral", "B-negative", "I-negative", "END"]

# Read training data
def read_training_data(path):
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

def calculate_number_of_labels(input_data):
    label_counts = {}
    for elem in input_data:
        label = elem[1]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return label_counts


def get_all_unique_tokens(input_data):
    unique_tokens = []
    for item in input_data:
        if item[0] not in unique_tokens:
            unique_tokens.append(item[0])
    return unique_tokens


def calculate_emission_parameters(data, all_unique_tokens, k=1.0):

    # Final index is for #UNK# tokens
   

    emission_counts = np.zeros((N, len(all_unique_tokens) + 1), dtype=np.longdouble)   #creates an NP array where rows - labels, columns - unique tokens
    emission_parameters = np.zeros((N, len(all_unique_tokens) + 1), dtype=np.longdouble)

    label_counts = []
    label_counts_dict = calculate_number_of_labels(data)    # a dictionary with the format : {label:count} , i.e, each key is a label which has its count as the corresponding value
    sorted_labels = sorted(label_counts_dict.keys())
    for label in sorted_labels:
        label_counts.append(label_counts_dict[label])
    label_counts = np.array(label_counts)


    for token, label in data:
        emission_counts[label - 1][all_unique_tokens.index(token)] += 1    #keep a count of how many times each token appears for each label in the data list
    # This is for the other case of #UNK# tokens
    emission_counts[:, -1] = [k] * N


    for index, _ in enumerate(emission_counts):
        emission_parameters[index] = emission_counts[index] / (label_counts[index] + k)


    return emission_parameters