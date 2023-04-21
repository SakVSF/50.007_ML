from math import log
import numpy as np
import pandas as pd
import sys



from part1 import Counts
from part2 import Missing, UniqVocab, Transition
no_parent = 0
no_parent_count = 0


#helper function
def calculate_emission(dict):
	'''
	Description:
	Input = "dict" storing emission counts
	Output =modified dictionary with emission probability
	'''
	for _, tag_to_word in dict.items():
		count = sum(tag_to_word.values())   #count(y)
		for tag, tagCount in tag_to_word.items():
			tag_to_word[tag] = tagCount / count    # count(y->x)/count(y)
	
	return dict

#helper function
def maxScores(i, maxScore, curr_tag, parent_tag, score):
	'''checks if index i is in score dictionary. 
	   If it is, updates the maxScore and parent tag for the given current tag. 
	   If it is not, create a dictionary for the current tag under index i
	   '''
	if i in score:
		score[i][curr_tag] = [maxScore, parent_tag]
	else:
		score[i] = {curr_tag: [maxScore, parent_tag]}



def emissionProbs(file):
	"""
	Description:
	emissionProbs, backwardProbs and forwardProbs are nested dictionaries of the format = {x(word): {y(tag): probability of x emitting y}}
	note this is different from HMM. Here emission probability is the probability of a word emitting a tag. In HMM it was a tag emitting a word

	Input = file 
	Output = emissionProbs, forwardProbs, backwardProbs, unique_tags(set of tags found in file)

	"""
	words = []
	tags = []
	emissionProbs = {}
	forwardProbs = {}
	backwardProbs = {}
	 
	
	with open(file, encoding="utf-8") as f:
		for line in f:
			sentence = line.strip()
			# if empty lines
			if len(sentence) == 0:
				continue
			# if contains word-tag pair
			else:
				space= sentence.rfind(" ")
				#word = sentence[:space]
				word = sentence[:space].lower()  #converts all words to lowercase 
				tag = sentence[space + 1:]          
				words.append(word)
				tags.append(tag)

	#calculating count(y->x)
	for i in range(0, len(words)):
		word = words[i]


		if i == 0: 
			#start of sentence
			prev_word = 'START'
		else:
			prev_word = words[i-1] 

		if i == len(words)-1:
			#end of sentence
			next_word = 'END'
		else:
			next_word = words[i+1]

		tag = tags[i]

		#updating emission counts for word, word-1, word+1 in respective dictionaries
		Counts(word, tag, emissionProbs)              
		Counts(prev_word, tag, forwardProbs)
		Counts(next_word, tag, backwardProbs)


	#calculating all emission probabilities 
	emissionProbs = calculate_emission(emissionProbs)
	forwardProbs = calculate_emission(forwardProbs)
	backwardProbs = calculate_emission(backwardProbs)

	'''
	If a word appears less than k times in the training set, it is replaced with #UNK#.
	The emission probabilities for the #UNK# token emitting a tag is normalized as=  count of a tag/  the total count of all tagstags.
	'''
	unique_tags = set(tags)

	#calculating emission counts for UNK token 

	tagCounts = {}   #dict format: { yi : count(yi)}
	for tag in tags:
		if tag in tagCounts:
			tagCounts[tag] += 1
		else:
			tagCounts[tag] = 1
	total_count = sum(tagCounts.values())   #totalcount
	for key, count in tagCounts.items():
		tagCounts[key] = count/total_count    # count(y)/totalcount

	#calculating emission probabilities for UNK token
	emissionProbs["#UNK#"] = tagCounts            # { "#UNK#"" : {yi : count(yi)/totalcount}}   
	forwardProbs["#UNK#"] = tagCounts
	backwardProbs["#UNK#"] = tagCounts

	return emissionProbs, forwardProbs, backwardProbs, unique_tags





def memmViterbi(emissionProbs, forwardProbs, backwardProbs, transitions, weights, vocab, tags, sentence):
	'''
	Decription:
	Defining and executing the discriminative/modified viterbi algorithm to help make a prediction
	
'''
	#score is a dictionary of format : {yi : {maximum probability , parent tag}}

	score = {}
	score[0] = {"_START": [0.0, None]}

	
	# forward algorithm
	for i in range(len(sentence)):
		word = sentence[i].lower()

		if i>1:
			prev_word = sentence[i-1].lower() 
		else:
			prev_word = 'START'  #beginning of sentence

		if i < len(sentence)-1 : 
			next_word = sentence[i+1].lower() 
		else:
			next_word = 'END'     #end of sentence
	

		#if word/prev_word/next_word not seen previously in training set, replace with #UNK#
		if word not in vocab:
			word = "#UNK#"
		if prev_word not in vocab:
			prev_word = "#UNK#"
		if next_word not in vocab:
			next_word = "#UNK#"
		

		for currenttag in tags:
			highScore = None
			parent= None

			if i == 0: #then previoustag only has 1 option: "_START":

				previousscore = score[0]["_START"]

				if Missing(currenttag, "_START", transitions) or \
					Missing(currenttag, word, emissionProbs): 

					#if tag not found in transition or emission dictionaries 
					maxScores(i+1, None, currenttag, None, score)   #set score and parentto None
				
				else:
					a = transitions["_START"][currenttag]
					b = emissionProbs[word][currenttag] 
					
					if Missing(currenttag, prev_word, forwardProbs):  #if no path from prev_word to current tag 
						forward = 1
					else: 
						forward = forwardProbs[prev_word][currenttag] 
					
					if Missing(currenttag, next_word, backwardProbs):   #if no path from current tag to next word 
						backward = 1
					else :
						backward = backwardProbs[next_word][currenttag] 

					#calculating the score
					currentscore = previousscore[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(forward) * weights[3] + log(backward) * weights[4] 
					
					highScore = currentscore
					parent= "_START"
					maxScores(i+1, highScore, currenttag, parent, score)

			else: #middle of sentence
				for previoustag in tags:	
					previousscore = score[i][previoustag]

					#if previoustag has no path to currenttag
					if previousscore[0] == None or Missing(currenttag, previoustag, transitions) or Missing(currenttag, word, emissionProbs):
						continue

					else:
						a = transitions[previoustag][currenttag]
						b = emissionProbs[word][currenttag] 
						
						if  Missing(currenttag, prev_word, forwardProbs): #if no path from prev_word to current tag 
							forward = 1
						else:
							forward = forwardProbs[prev_word][currenttag]

						
						if Missing(currenttag, next_word, backwardProbs): #if no path from current tag to next word 
							backward =1 
						else: 
							backward = backwardProbs[next_word][currenttag] 

						#calculating the score 
						currentscore= previousscore[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(forward) * weights[3] + log(backward) * weights[4] 
						
						if highScore is None or currentscore> highScore:
							highScore = currentscore
							parent= previoustag

				if highScore is None:
					#if no possible paths from the previous tag -> no score and no parent
					maxScores(i+1, None, currenttag, None, score)
				else:
					maxScores(i+1, highScore, currenttag, parent, score)
			
	# final iteration stop case
	highScore = None
	parent= None
	i = len(sentence)
	for previoustag in tags:	
		previousscore = score[i][previoustag]
		
		if previousscore[0] == None or Missing("_STOP", previoustag, transitions):
			# if no path from previous tag to STOP
			continue
		else:
			previousscore = score[i][previoustag]
			a = transitions[previoustag]["_STOP"]

			
			if Missing(currenttag, prev_word, forwardProbs):  #if no path from prev_word to currenttag
				forward = 1
			else:  
				forward = forwardProbs[prev_word][currenttag]

			#note that no emission probability or backward probability is calculated since it is stop case.
			currentscore = previousscore[0] * weights[0] + log(a) * weights[1] + log(forward) * weights[3]
			if highScore is None or currentscore > highScore:
				highScore = currentscore
				parent= previoustag
	
	if highScore is None:
		#if  no possible paths to _STOP		
		maxScores(i+1, None, "_STOP", None, score)
	else:
		maxScores(i+1, highScore, "_STOP", parent, score)
	


	#backpropagation
	prediction = []
	currenttag = "_STOP"	

	for i in range(len(sentence)+1, 0, -1): #back to front
		parent= score[i][currenttag][1]

		if parent== None:
			global no_parent
			no_parent += 1			
			
			highScore = None
			bestParent = None

			for pair in list(score[i-1].keys()):
				score_parent = score[i-1][pair]
				temp_score = score_parent[0]
				temp_parent = score_parent[1]

				if temp_parent== None or temp_score== None:
					continue
				else:
					if highScore == None or temp_score > highScore:
						highScore = temp_score 
						bestParent = temp_parent

			if bestParent== None:
				global no_parent_count
				no_parent_count += 1
				if list(score[i-1].keys())[0] == "_START":
					parent= "_START"
				else:
					parent= 'O' #defaults to O if no parent because O is the most common tag
			
			else:
				parent= bestParent
		
		#reached end 
		if parent== "_START":
			break
			
		
		prediction.append(parent)
		currenttag = parent 

	prediction.reverse()
	return prediction


def ViterbiLoop(emissionProbs, forwardProbs, backwardProbs, transitions, weights, vocab, tags,  inputFile, outputFile):

	with open(inputFile) as inp, open(outputFile, "w", encoding="UTF-8") as out:
		sentence = []
		for line in inp:
			#extract sentence 
			if line != "\n":
				word = line.strip()
				sentence.append(word)

			#end of sentence reached. predict tags for sentence and write to output file
			else:
				sequence = memmViterbi(emissionProbs, forwardProbs, backwardProbs, transitions, weights, vocab, tags,  sentence)
				
				for i in range(len(sequence)):
					out.write("{} {}\n".format(sentence[i], sequence[i]))
				out.write("\n")
				sentence = []

	print("prediction completed")




def mem(dir, weights):
	
	file = open(dir + "/train", "r", encoding="utf-8")
	lines_in_pairs = [line.rstrip('\n').split(" ") for line in file]

	word_tag_pairs = [['__Start__', "Start"]]  # First Start tag

	for pair in lines_in_pairs:
		if len(pair) == 1:  #add stop and start tags for every empty line (new sequence)
			word_tag_pairs.append(['__Stop__', "Stop"])
			word_tag_pairs.append(['__Start__', "Start"])
		else:
			word_tag_pairs.append(pair)

	word_tag_pairs = word_tag_pairs[:-1]   #remove START tag at the end 
	word_tag_df = pd.DataFrame(word_tag_pairs)  #create dataframe  
	tags = word_tag_df[1].unique()              #collection of all tags
	vocab = word_tag_df[0].unique()             #collection of all words 
	#tagc_dict = word_tag_df[1].value_counts().to_dict()      
	tags = np.sort(tags)                       
	tags = tags[::-1]   #sorting tags in descending order
 

	emissions, forwardProbs, backwardProbs, tags = emissionProbs(dir + '/train')

	transitions = Transition(dir + '/train')
	vocab = UniqVocab(dir + '/train')
	ViterbiLoop(emissions, forwardProbs, backwardProbs, transitions, weights, vocab, tags, dir +'/dev.in', dir + '/dev.p4.out')
	#run on test set 

	ViterbiLoop(emissions, forwardProbs, backwardProbs, transitions, weights, vocab, tags, dir +'/test.in', dir + '/test.p4.out')
		
	

def main(args):
	data = ["EN", "FR"]
	#weights_FR= [1, 1, 7, 0, 0.1] #best combination for FR dataset - 0.5286, 0.3597
	weights_FR = [1, -1, 6, 0, 0]
	weights_EN= [1.2, 3.35, 6, 3, 1 ]  #best combination for EN dataset 
	
	if args in data:
		if args == "EN":
			mem(args, weights_EN)
		else:
			mem(args, weights_FR)

	else:
		print("Argument must be either EN or FR. Please try again")

	





if __name__ == "__main__":
    args = sys.argv
    main(args[1])