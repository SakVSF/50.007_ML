from part1 import getEmissions, addCount
import sys
import time
from pathlib import Path
from math import log


def getTransitions(file): 
	'''
	transitions is a nested dictionary of the dict format -> { yi-1 : {yi : probability of transition fron yi-1 to yi}}
	'''
	start = "_START"
	stop = "_STOP" 
	transitions = {}   
	count = {start: 0}
	prev_tag = start
	with open(file, encoding="utf-8") as f:
		for line in f:
			sentence = line.strip()
			# sentence has ended
			if len(sentence) == 0:
				addCount(prev_tag, stop, transitions)   
				prev_tag = start                         #set prev_tag to START as starting a new sentence in next iteration

			# in the middle of a sentence
			else:
				space = sentence.rfind(" ")
				curr_tag = sentence[space + 1:]

				# A new sentence, update count(START)
				if prev_tag == start:
					count[start] += 1                    

				# update count(y)
				if curr_tag in count:
					count[curr_tag] += 1
				else:
					count[curr_tag] = 1
				
				# update count(prev, curr)
				addCount(prev_tag, curr_tag, transitions)

				prev_tag = curr_tag


		#reached end of file 
		# add count(prev, stop) if no blank lines at EOF
		if prev_tag != start:
			addCount(prev_tag, stop, transitions)
			prev_tag = start

	# Calculate transition probabilities 
	for prev_tag, currDict in transitions.items():   
		for curr_tag, currCount in currDict.items():
			currDict[curr_tag] = currCount /( float(count[prev_tag]))   # count(yi−1, yi) / count(y-)

	return transitions



def convert(file):
    # convert train file to set of unique vocab
    out = set()
    with open(file, encoding="utf-8") as f:
        for line in f:
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0:
                continue
            else:
                last_space_index = temp.rfind(" ")
                word = temp[:last_space_index].lower()
                out.add(word)

    return out


def isMissing(child, parent, hashmap):
    # check whether child's parent is parent in given dictionary
    return (child not in hashmap[parent]) \
        or (hashmap[parent][child] == 0)


def viterbiAlgo(emissions, transitions, vocab, lines):

    tags = emissions.keys()
    score = {}
    score[0] = {"_START": [0.0, None]}

    # forward algorithm
    for i in range(1, len(lines) + 1):
        word = lines[i - 1].lower()

        # Replace word with #UNK# if not in train
        if word not in vocab:
            word = "#UNK#"

        for currTag in tags:
            highScore = None
            parent = None

            # Check that word can be emitted from currTag
            if isMissing(word, currTag, emissions):
                continue

            b = emissions[currTag][word]

            for prevTag, prevScore in score[i - 1].items():

                # Check that currTag can transit from prevTag and prevPie exist
                if isMissing(currTag, prevTag, transitions) or \
                        prevScore[0] is None:
                    continue

                a = transitions[prevTag][currTag]

                # Calculate score
                currScore = prevScore[0] + log(a) + log(b)

                if highScore is None or currScore > highScore:
                    highScore = currScore
                    parent = prevTag

            # Update score
            if i in score:
                score[i][currTag] = [highScore, parent]
            else:
                score[i] = {currTag: [highScore, parent]}

    # Final iteration stop case
    highScore = None
    parent = None

    for prevTag, prevScore in score[len(lines)].items():
        # Check prev can lead to a stop
        if "_STOP" in transitions[prevTag]:
            a = transitions[prevTag]["_STOP"]
            if a == 0 or prevScore[0] is None:
                continue

            currScore = prevScore[0] + log(a)
            if highScore is None or currScore > highScore:
                highScore = currScore
                parent = prevTag

    score[len(lines) + 1] = {"_STOP": [highScore, parent]}

    # backtracking to get prediction
    prediction = []
    curr = "_STOP"
    i = len(lines)

    while True:
        parent = score[i + 1][curr][1]
        if parent is None:
            #print(i)
            parent = list(score[i].keys())[0]

        if parent == "_START":
            break

        prediction.append(parent)
        curr = parent
        i -= 1

    prediction.reverse()
    return prediction


def predictWithViterbi(emissions, transitions, vocab, inputFile, outputFile):
    with open(inputFile, encoding="utf-8") as f, open(outputFile, "w", encoding="utf-8") as out:
        sentence = []

        for line in f:
            # form sentence
            if line != "\n":
                word = line.strip()
                sentence.append(word)

            # predict tag sequence
            else:
                sequence = viterbiAlgo(emissions, transitions, vocab, sentence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []
    print("Prediction Done!")


def main(args):
    data = ["EN", "FR"]
    if args in data:
        dir = Path(args)
        start = time.time()
        emissions = getEmissions(dir/'train')
        transitions = getTransitions(dir/'train')
        vocab = convert(dir/'train')
        predictWithViterbi(emissions, transitions, vocab,
                           dir/'dev.in', dir/'dev.p2.out')
        end = time.time()
        print(f"Elapsed time: {round(end-start,2)}s")
    else:
        print("Specified Dataset must be either EN, SG or CN, Run again...")


if __name__ == "__main__":
    args = sys.argv
    main(args[1])
