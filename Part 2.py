# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:50:27 2023

@author: User
"""

import math
import numpy as np
import random
from collections import Counter

def estimate_transition_parameters(training_set):
    """
    This function estimates the transition parameters from the training set using MLE.

    Arguments: Training_set (list): A list of sentences, where each sentence is a list of (word, tag) tuples.

    Returns dict: A dictionary containing the transition parameters, where the keys are (tag1, tag2) tuples and
    the values are the estimated transition probabilities.
    
    Instructions to test models: 1) To run after part 1 to give the dev.p1.out files
                                 2) Place evalResult.py script in the same directory as your EN and FR dev.out and dev.prediction files.
                                 3) In the terminal or command prompt, navigate to the directory where the files are located.
                                 4) Run the following command: 'python evalResult.py dev.out dev.prediction'
                                 5) It wil calculate precision, recall, and F-score. So it will test both the Viterbi algorithm and the Python function
                                    that estimates the transition parameters.
    """
    # Counting the number of times each tag occurs in the training set
    tag_counts = {}
    for sentence in training_set:
        for i in range(len(sentence)):
            tag = sentence[i][1]
            if tag not in tag_counts:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1
    
    # Counting the number of times each tag occurs after each previous tag in the training set
    transition_counts = {}
    for sentence in training_set:
        for i in range(1, len(sentence)):
            prev_tag = sentence[i-1][1]
            curr_tag = sentence[i][1]
            transition = (prev_tag, curr_tag)
            if transition not in transition_counts:
                transition_counts[transition] = 1
            else:
                transition_counts[transition] += 1
    
    # Calculating the estimated transition probabilities
    transition_probs = {}
    for transition, count in transition_counts.items():
        prev_tag = transition[0]
        if prev_tag == "STOP":
            continue
        denominator = tag_counts[prev_tag]
        if denominator == 0:
            transition_probs[transition] = 0
        else:
            transition_probs[transition] = count / denominator
    
    # Calculating the special case transition probabilities
    denominator = tag_counts["START"]
    if denominator > 0:
        transition_probs[("START", "O")] = 0
        for tag in tag_counts.keys():
            transition_probs[("START", tag)] = 0
    
    denominator = tag_counts["O"]
    if denominator > 0:
        transition_probs[("O", "STOP")] = 0
        for tag in tag_counts.keys():
            transition_probs[(tag, "STOP")] = 0
    
    return transition_probs

def viterbi(sentence, states, start_prob, trans_prob, emit_prob):
    """
    Runs the Viterbi algorithm on a sentence with the given model parameters.

    Arguments: sentence (list): A list of words in the sentence.
               states (list): A list of possible tags.
               start_prob (dict): A dictionary containing the probabilities of each tag starting a sentence.
               trans_prob (dict): A dictionary containing the transition probabilities between each pair of tags.
               emit_prob (dict): A dictionary containing the emission probabilities of each word for each tag.
    Returns list: A list of the most likely tags for the words in the sentence.
    Instructions to test models: 1) To run after part 1 to give the dev.p1.out files
                                 2) Place evalResult.py script in the same directory as your EN and FR dev.out and dev.prediction files.
                                 3) In the terminal or command prompt, navigate to the directory where the files are located.
                                 4) Run the following command: 'python evalResult.py dev.out dev.prediction'
                                 5) It wil calculate precision, recall, and F-score. So it will test both the Viterbi algorithm and the Python function
                                    that estimates the transition parameters.
    """
    # Initializing the trellis
    trellis = [{}]
    for state in states:
        if start_prob[state] > 0 and emit_prob.get((sentence[0], state)) is not None:
            trellis[0][state] = {"prob": start_prob[state] * emit_prob[(sentence[0], state)], "prev": None}
    
    # Filling in the rest of the trellis
    for i in range(1, len(sentence)):
        trellis.append({})
        for state in states:
            max_prob = 0
            max_prev = None
            for prev_state in trellis[i-1].keys():
                trans_p = trans_prob.get((prev_state, state), 0)
                emit_p = emit_prob.get((sentence[i], state), 0)
                prob = trellis[i-1][prev_state]["prob"] * trans_p * emit_p
                if prob > max_prob:
                    max_prob = prob
                    max_prev = prev_state
            if max_prob > 0:
                trellis[i][state] = {"prob": max_prob, "prev": max_prev}
    
    # Finding the most likely tag sequence by backtracking through the trellis
    max_prob = 0
    max_state = None
    for state in trellis[-1].keys():
        prob = trellis[-1][state]["prob"]
        if prob > max_prob:
            max_prob = prob
            max_state = state
    if max_state is None:
        return ["O"] * len(sentence)
    
    max_tag_seq = [max_state]
    for i in range(len(sentence)-2, -1, -1):
        max_tag_seq.insert(0, trellis[i+1][max_tag_seq[0]]["prev"])
    
    return max_tag_seq