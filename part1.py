import sys
import time
from pathlib import Path

def Counts(parent, child, d):
    """
    Description:
    Defining and incrementing the count of [parent][child] in a dictionary d
    """
    if parent in d:
        if child in d[parent]:
            d[parent][child] += 1
        else:
            d[parent][child] = 1
    else:
        d[parent] = {child: 1}

def Emission(file, k=0.5):
    """
    Description:
    input file = dev.in
    output = emission parameters (dict)
    Words that appear less than k times will be replaced with #UNK#
    dictionary format = {i: {o:emission prob}}
    """
    emission = {}
    count = {}
    with open(file, encoding="utf-8") as f:
        for line in f:
            temp = line.strip()
            if len(temp) == 0:
                continue
            else:
                last_space_index = temp.rfind(" ")
                x = temp[:last_space_index].lower()
                y = temp[last_space_index + 1:]

                # updating count[y]
                if y in count:
                    count[y] += 1
                else:
                    count[y] = 1

                # updating count(y->x)
                Counts(y, x, emission)
    #count = {tag1: count1, tag2: count2, etc}
    #emission = {tag: {word1:count1, word2:count2, etc.}}
    #converting the counts to emission probabilities
    for y, x_Dictionary in emission.items():
        for x, x_Count in x_Dictionary.items():
            x_Dictionary[x] = x_Count / float(count[y] + k)
        #replacing with a UNK variable
        emission[y]["#UNK#"] = k / float(count[y] + k)
    return emission

def PredictionSentiments(emission, testfile, outputfile):
    """
    Description:
    predicts sequence labels using argmax(emission)
    # finds the best #UNK# for later use
    """
    unktag = "O"
    unkP = 0
    for tag in emission.keys():
        if emission[tag]["#UNK#"] > unkP:
            unktag = tag
    with open(testfile, encoding="utf-8") as f, open(outputfile, "w", encoding="utf-8") as out:
        for line in f:
            if line == "\n":
                out.write(line)
            else:
                word = line.strip().lower()
                #The code below finds the highest probability for each word
                bestprobability = 0
                besttag = ""
                for tag in emission:
                    if word in emission[tag]:
                        if emission[tag][word] > bestprobability:
                            bestprobability = emission[tag][word]
                            besttag = tag

                if besttag == "":
                    besttag = unktag

                out.write("{} {}\n".format(word, besttag))
    print("Prediction is generated successfully")

def main(args):
    """
    Description:
    Inputting of the necessary files and the defining of the output file dev.p1.out
    """
    datasets = ["EN", "FR"]
    if args in datasets:
        dir = Path(args)
        start = time.time()
        emission = Emission(dir/'train')
        PredictionSentiments(emission, dir/'dev.in', dir/'dev.p1.out')
        end = time.time()
        print(f"Time taken: {round(end-start,2)}s")
    else:
        print("The dataset to be entered must either be EN or FR, please try again.")

if __name__ == "__main__":
    args = sys.argv
    main(args[1])
