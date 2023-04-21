# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:17:16 2023

@author: User
"""
import time
import sys
from pathlib import Path
from math import log
from part1 import Emission

def get_ngrams(words, n):
    """
    Extracts n-grams from a list of words
    """
    if len(words) >= n:
        ngrams = list(zip(*[words[i:] for i in range(n)]))
    else:
        ngrams = [None]*(n-1-len(words)) + words
    return ngrams


def Counts(parent1, parent2, child, d):
    """
    Description:
    Defining and incrementing the count of [parent2, parent1][child] in a dictionary d
    """
    if (parent1, parent2) in d:
        if child in d[(parent1, parent2)]:
            d[(parent1, parent2)][child] += 1
        else:
            d[(parent1, parent2)][child] = 1
    else:
        d[(parent1, parent2)] = {child: 1}



def Transition(data):
    """
    Description:
    Computing transition matrix
    """
    # Initialize variables
    count = {}
    currentdict = {}

    # Read training data
    with open(data, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Get previous2, previous1, and current words
                words = line.split()
                previous2, previous1, current, tag = get_ngrams(words, 5)

                # Update count dictionary
                if previous2:
                    Counts(previous2, previous1, current, count)  # change here
                else:
                    Counts(previous1, current, count)

                # Update current word count
                if current in currentdict:
                    currentdict[current] += 1
                else:
                    currentdict[current] = 1

    # Compute transition matrix
    transition = {}
    for previous2_previous1 in count:
        previous2, previous1 = previous2_previous1
        currentdict = count[previous2_previous1]
        total = float(sum(currentdict.values()))
        transition[previous2_previous1] = {current: currentdict[current]/total for current in currentdict}

    return transition


def UniqVocab(file):
    """
    Description:
    converting the training file into 'vocab's
    """
    out = set()
    with open(file, encoding="utf-8") as f:
        for line in f:
            temp = line.strip()

            #ignoring empty lines
            if len(temp) == 0:
                continue
            else:
                lastindex = temp.rfind(" ")
                word = temp[:lastindex].lower()
                out.add(word)

    return out

def Missing(child, parent, hashmap):
    """
    Description:
    To check whether a child's parent is actually the parent in the given dictionary
    """
    return (child not in hashmap[parent]) \
        or (hashmap[parent][child] == 0)

def Viterbi(emission, transition, vocab, lines):
    """
    Description:
    Defining and executing the viterbi algorithm to help make a prediction
    """
    tag = emission.keys()
    score = {}
    score[0] = {"_START": [0.0, None]}
    for i in range(1, len(lines) + 1):
        words = lines[i - 1].lower()
        
        #Replace 'words' with variable #UNK# if not present in 'train' file
        if words not in vocab:
            words = "#UNK#"
        for currenttag in tag:
            highscore = None
            parent = None

            #Checking whether words can be emitted from currenttag
            if Missing(words, currenttag, emission):
                continue
            b = emission[currenttag][words]
            for previoustag, previousscore in score[i - 1].items():

                #Checking that currenttag can transmit from previoustag and that previos pie exists
                if Missing(currenttag, previoustag, transition) or \
                        previousscore[0] is None:
                    continue

                a = transition[previoustag][currenttag]

                #Calculating the score
                currentscore = previousscore[0] + log(a) + log(b)
                if highscore is None or currentscore > highscore:
                    highscore = currentscore
                    parent = previoustag

            # Update score
            if i in score:
                score[i][currenttag] = [highscore, parent]
            else:
                score[i] = {currenttag: [highscore, parent]}

    # Final iteration stop case
    highscore = None
    parent = None

    for previoustag, previousscore in score[len(lines)].items():
        #Checking if previous can lead to a stop
        if "_STOP" in transition[previoustag]:
            a = transition[previoustag]["_STOP"]
            if a == 0 or previousscore[0] is None:
                continue
            currentscore = previousscore[0] + log(a)
            if highscore is None or currentscore > highscore:
                highscore = currentscore
                parent = previoustag
    score[len(lines) + 1] = {"_STOP": [highscore, parent]}

    #Attempting to backtrack to get a prediction
    prediction = []
    current = "_STOP"
    i = len(lines)

    while True:
        parent = score[i + 1][current][1]
        if parent is None:
            #print(i)
            parent = list(score[i].keys())[0]

        if parent == "_START":
            break

        prediction.append(parent)
        current = parent
        i -= 1

    prediction.reverse()
    return prediction

def ViterbiPrediction(emission, transition, vocab, inputFile, outputFile):
    """
    Description:
    To run the viterbi algorithm with the previously defined variables and generate a prediction
    """
    with open(inputFile, encoding="utf-8") as f, open(outputFile, "w", encoding="utf-8") as out:
        sentence = []
        for line in f:
            #To form sentences
            if line != "\n":
                words = line.strip()
                sentence.append(words)
            #For prediction of the tag sequence
            else:
                sequence = Viterbi(emission, transition, vocab, sentence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []
    print("Prediction is generated successfully")

def main(args):
    """
    Description:
    Inputting of the necessary files and the defining of the output file dev.p2.out
    """
    data = ["EN", "FR"]
    if args in data:
        dir = Path(args)
        start = time.time()
        emission = Emission(dir/'train')
        transition = Transition(dir/'train')
        vocab = UniqVocab(dir/'train')
        ViterbiPrediction(emission, transition, vocab,
                           dir/'dev.in', dir/'dev.p2.out')
        end = time.time()
        print(f"Time taken: {round(end-start,2)}s")
    else:
        print("The dataset to be entered must either be EN or FR, please try again.")

if __name__ == "__main__":
    args = sys.argv
    main(args[1])



"""import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the data
train_file = {"EN":"C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/EN/train", "FR": "C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/FR/train"}
dev_file = {"EN": "C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/EN/dev.in", "FR": "C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/FR/dev.in"}
test_file = {"EN": "C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/EN/dev.out", "FR": "C:/Users/User/Downloads/ml-sequence-labelling-main/ml-sequence-labelling-main/FR/dev.out"}


train_data = {}
for lang in ["EN", "FR"]:
    train_data[lang] = []
    with open(train_file[lang], "r") as f:
        sentences = f.read().strip().split("\n\n")
        for sentence in sentences:
            words = sentence.split("\n")
            for word in words:
                print(word)

dev_data = {}
for lang in ["EN", "FR"]:
    dev_data[lang] = []
    with open(dev_file[lang], "r") as f:
        sentences = f.read().strip().split("\n\n")
        for sentence in sentences:
            words = sentence.split("\n")
            for word in words:
                print(word)

test_data = {}
for lang in ["EN", "FR"]:
    test_data[lang] = []
    with open(test_file[lang], "r") as f:
        sentences = f.read().strip().split("\n\n")
        for sentence in sentences:
            words = sentence.split("\n")
            for word in words:
                print(word)


# Define the model parameters
N = 48  # number of states
M = 11971  # number of unique words
pi = np.ones(N) / N  # initial state distribution
transition_prob = np.zeros((N, N, N))  # transition probabilities
emission_prob = np.zeros((N, M))  # emission probabilities

tag2id = {
    "O": 0,
    "B-LOC": 1,
    "I-LOC": 2,
    "B-MISC": 3,
    "I-MISC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-PER": 7,
    "I-PER": 8,
}

id2tag = {v: k for k, v in tag2id.items()}

word2id = {}
id2word = {}
for lang in ["EN", "FR"]:
    for sentence in train_data[lang]:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[word2id[word]] = word
                
# Learn the model parameters with train
for lang in ["EN", "FR"]:
    for sentence in train_data[lang]:
        for i in range(len(sentence)):
            word, tag = sentence[i].rsplit("/", 1)
            emission_prob[tag2id[tag], word2id[word]] += 1
            if i >= 2:
                transition_prob[tag2id[sentence[i-2].rsplit("/", 1)[0]], tag2id[sentence[i-1].rsplit("/", 1)[0]], tag2id[tag]] += 1

transition_prob /= transition_prob.sum(axis=2, keepdims=True)
emission_prob /= emission_prob.sum(axis=1, keepdims=True)

# Viterbi algorithm for decoding
for lang in ["EN", "FR"]:
    with open(dev_file[lang], "r") as f:
        with open("dev.p3.out." + lang, "w") as fout:
            for sentence in dev_data[lang]:
                T = len(sentence)
                dp = np.zeros((N, N, T))
                backpointers = np.zeros((N, N, T), dtype=int)

                # Initialization
                for i in range(N):
                    for j in range(N):
                        dp[i, j, 0] = np.log(pi[i]) + np.log(transition_prob[j, i, :]).sum() + np.log(emission_prob[i, word2id.get(sentence[0], 0, M)] or 1e-10)

                # Forward recursion
                for t in range(2, T):
                    for i in range(N):
                        for j in range(N):
                            for k in range(N):
                                if dp[j, k, t-1] == -np.inf or transition_prob[k, j, :].sum() == 0:
                                    continue
                                score = dp[j, k, t-1] + np.log(transition_prob[k, j, i]) + np.log(emission_prob[i, word2id.get(sentence[t], 0, M)] or 1e-10)
                                if score > dp[k, i, t]:
                                    dp[k, i,t] = score
                                    backpointers[k, i, t] = j

                tagset = sorted(list(set([tag for sentence in train_data[lang] for _, tag in sentence])))
                T = len(tagset)
                # Backward recursion
                best_path = [0] * T
                best_score = -np.inf
                for i in range(N):
                    for j in range(N):
                        if dp[i, j, T-1] > best_score:
                            best_score = dp[i, j, T-1]
                            best_path[T-1] = i
                            best_path[T-2] = j
                for t in range(T-3, -1, -1):
                    best_path[t] = backpointers[best_path[t+1], best_path[t+2], t+2]

                # Output the prediction
                for i in range(T):
                    print(sentence[i] + "/" + id2tag[best_path[i]] + " ", end="", file=fout)
                print("", file=fout)

"""


