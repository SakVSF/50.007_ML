import time
import sys
from pathlib import Path
from math import log
from part1 import Emission, Counts

def Transition(file):
    """
    Description:
    Input = 'file', Output = transition parameters
    Also return Dict: {y_i-1: {y_i: transition}}
    """
    start = "_START"
    stop = "_STOP"
    transition = {}
    count = {start: 0}
    previous = start
    with open(file, encoding="utf-8") as f:
        for line in f:
            temp = line.strip()
            if len(temp) == 0:
                Counts(previous, stop, transition)
                previous = start
            else:
                last_index = temp.rfind(" ")
                current = temp[last_index + 1:]
                #updating count(start) upon finding new sentence
                if previous == start:
                    count[start] += 1
                #updating count(y)
                if current in count:
                    count[current] += 1
                else:
                    count[current] = 1
                Counts(previous, current, transition)
                previous = current

        #adding up count(previous, stop) if no blank lines are found at the end of the file
        if previous != start:
            Counts(previous, stop, transition)
            previous = start

    #converting counts to transitions
    for previous, currentdict in transition.items():
        for current, currentcount in currentdict.items():
            currentdict[current] = currentcount / float(count[previous])
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
