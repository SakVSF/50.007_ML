from math import log
#from evalResult import eval
import numpy as np
import pandas as pd
import copy
import sys
import time
from pathlib import Path

from part1 import addCount
from part2 import isMissing, convert, getTransitions
_parentless_stop = 0
_deep_parentless_count = 0



def calculate_emission(dict):
	for _, tagCountDict in dict.items():
		count = sum(tagCountDict.values())   #count(y)
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count    # count(y->x)/count(y)
	
	return dict


def discriminative_emissions(file, k=1):
	"""
	emissions, ackward_emissions and forward_emissions are nested dictionaries of the format = {x(word): {y(tag): probability of x emitting y}}
	note this is different from HMM. Here emission probability is the probability of a word emitting a tag. In HMM it was a tag emitting a word
	"""
	emissions = {}
	forward_emissions = {}
	backward_emissions = {}
	 
	words = []
	tags = []
	with open(file, encoding="utf-8") as f:
		for line in f:
			sentence = line.strip()
			# if empty lines
			if len(sentence) == 0:
				continue
			# if contains word-tag pair
			else:
				space= sentence.rfind(" ")
				word = sentence[:space].lower()  #convert all words to lowercase 
				tag = sentence[space + 1:]          
				words.append(word)
				tags.append(tag)

	#calculating count(y->x)
	for i in range(0, len(words)):
		word = words[i]

		if i == 0: 
			prev_word = 'START'
		else:
			
			prev_word = words[i-1] 

		if i == len(words)-1:
			next_word = 'END'
		else:
			next_word = words[i+1]

		tag = tags[i]

		#incrementing count of emission for word, word-1, word+1 
		addCount(word, tag, emissions)              
		addCount(prev_word, tag, forward_emissions)
		addCount(next_word, tag, backward_emissions)


	#calculating emission probabilities 

	emissions = calculate_emission(emissions)
	forward_emissions = calculate_emission(forward_emissions)
	backward_emissions = calculate_emission(backward_emissions)

	'''
	If a word appears less than k times in the training set, it is replaced with #UNK#.
	The emission probabilities for the #UNK# token emitting a tag is normalized as=  count of a tag/  the total count of all tagstags.
	'''
	unique_tags = set(tags)

	#calculating emission probabilities for UNK token 

	tag_counts = {}   #dict format: { yi : count(yi)}
	for tag in tags:
		if tag in tag_counts:
			tag_counts[tag] += 1
		else:
			tag_counts[tag] = 1
	total_count = sum(tag_counts.values())   #totalcount
	for key, count in tag_counts.items():
		tag_counts[key] = count/total_count    # count(y)/totalcount

	emissions["#UNK#"] = tag_counts            # { "#UNK#"" : {yi : count(yi)/totalcount}}   
	forward_emissions["#UNK#"] = tag_counts
	backward_emissions["#UNK#"] = tag_counts

	return emissions, forward_emissions, backward_emissions, unique_tags




def setHighscores(i, highscore, curr_tag, parent_tag, score):
	'''checks if i is in highscore dictionary. 
	   If it is, updates the highscore and parent tag for the given current tag. 
	   If it is not, create a dictionary for the current tag under i 
	   '''
	if i in score:
		score[i][curr_tag] = [highscore, parent_tag]
	else:
		score[i] = {curr_tag: [highscore, parent_tag]}



def discriminativeViterbiAlgo(emissions, forward_emissions, backward_emissions, transitions, weights, vocab, tags, sentence):
	'''
	highscores is a dictionary of format : {yi : {maximum probability , parent tag}}

'''
	highscores = {}
	highscores[0] = {"_START": [0.0, None]}

	
	# forward algorithm
	for i in range(len(sentence)):
		word = sentence[i].lower()

		if i>1:
			prev_word = sentence[i-1].lower() 
		else:
			prev_word = 'START'

		if i < len(sentence)-1 : 
			next_word = sentence[i+1].lower() 
		else:
			next_word = 'END'
	

		#if word/prev_word/next_word not seen previously in training set, replace with #UNK#
		if word not in vocab:
			word = "#UNK#"
		if prev_word not in vocab:
			prev_word = "#UNK#"
		if next_word not in vocab:
			next_word = "#UNK#"
		

		for currTag in tags:
			highScore = None
			parentTag = None

			if i == 0: #then prevTag only has 1 option: "_START":
				prevScoreParentPair = highscores[0]["_START"]
				if isMissing(currTag, "_START", transitions) or isMissing(currTag, word, emissions):
					setHighscores(i+1, None, currTag, None, highscores)
				else:
					a = transitions["_START"][currTag]
					b = emissions[word][currTag] 
					b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
					b_backward = backward_emissions[next_word][currTag] if not isMissing(currTag, next_word, backward_emissions) else 1

					tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(b_forward) * weights[3] + log(b_backward) * weights[4] #+ log(b_forward2) * weights[5] + log(b_backward2) * weights[6]
					
					highScore = tempScore
					parentTag = "_START"
					setHighscores(i+1, highScore, currTag, parentTag, highscores)

			else: #prevTags can be any of the available tags
				for prevTag in tags:	
					prevScoreParentPair = highscores[i][prevTag]
					# if prev node is disjointed, aka no score
					if prevScoreParentPair[0] == None or isMissing(currTag, prevTag, transitions) or isMissing(currTag, word, emissions):
						#then this prevTag has no path to currTag
						continue
					else:
						a = transitions[prevTag][currTag]
						b = emissions[word][currTag] # if not isMissing(currTag, word, emissions) else 1
						b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
						b_backward = backward_emissions[next_word][currTag] if not isMissing(currTag, next_word, backward_emissions) else 1

						tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(b_forward) * weights[3] + log(b_backward) * weights[4] 
						
						if highScore is None or tempScore > highScore:
							highScore = tempScore
							parentTag = prevTag

				if highScore is None:
					#if even after iterating through all possibilities the highscore is none, this means there were no possible paths from the previous node, and so we set this node as disjointed
					#disjointed means, no score and no parent
					setHighscores(i+1, None, currTag, None, highscores)
				else:
					setHighscores(i+1, highScore, currTag, parentTag, highscores)
			
	# _STOP case
	highScore = None
	parentTag = None
	i = len(sentence)
	for prevTag in tags:	
		prevScoreParentPair = highscores[i][prevTag]
		# if prev node is disjointed, aka no score
		if prevScoreParentPair[0] == None or isMissing("_STOP", prevTag, transitions):
			continue
		else:
			prevScoreParentPair = highscores[i][prevTag]
			a = transitions[prevTag]["_STOP"]
			b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
			tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b_forward) * weights[3]
			if highScore is None or tempScore > highScore:
				highScore = tempScore
				parentTag = prevTag
	
	if highScore is None:
		#this means there are no possible paths to _STOP		
		setHighscores(i+1, None, "_STOP", None, highscores)
	else:
		setHighscores(i+1, highScore, "_STOP", parentTag, highscores)
	


	#backpropagation
	prediction = []
	currTag = "_STOP"	
	for i in range(len(sentence)+1, 0, -1): #back to front
		parentTag = highscores[i][currTag][1]
		if parentTag == None:
			global _parentless_stop
			_parentless_stop += 1			
			#this is a disjointed sentence
			#lets choose a parent that has a parent
			candidateHighscore = None
			bestParentCandidateTag = None
			for candidateParentTag in list(highscores[i-1].keys()):
				candidateScoreParentPair = highscores[i-1][candidateParentTag]
				if candidateScoreParentPair[1] == None or candidateScoreParentPair[0] == None:
					continue
				else:
					if candidateHighscore == None or candidateScoreParentPair[0] > candidateHighscore:
						candidateHighscore = candidateScoreParentPair[0]
						bestParentCandidateTag = candidateParentTag

			if bestParentCandidateTag == None:
				global _deep_parentless_count
				_deep_parentless_count += 1
				if list(highscores[i-1].keys())[0] == "_START":
					parentTag = "_START"
				else:
					parentTag = 'O' #defaults to O if no parent because O is the most common tag
			else:
				parentTag = bestParentCandidateTag
		
		if parentTag == "_START":
			break
			
		# print(currTag, parentTag)
		prediction.append(parentTag)
		currTag = parentTag

	prediction.reverse()
	return prediction


def ViterbiLoop(emissions, forward_emissions, backward_emissions, transitions, weights, vocab, tags,  inputFile, outputFile):

	with open(inputFile) as inp, open(outputFile, "w", encoding="UTF-8") as out:
		sentence = []
		for line in inp:
			#extract sentence 
			if line != "\n":
				word = line.strip()
				sentence.append(word)

			#end of sentence reached. predict tags for sentence and write to output file
			else:
				sequence = discriminativeViterbiAlgo(emissions, forward_emissions, backward_emissions, transitions, weights, vocab, tags,  sentence)
				
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
 

	emissions, forward_emissions, backward_emissions, tags = discriminative_emissions(dir + '/train')

	transitions = getTransitions(dir + '/train')
	vocab = convert(dir + '/train')
	ViterbiLoop(emissions, forward_emissions, backward_emissions, transitions, weights, vocab, tags, dir +'/dev.in', dir + '/dev.p4.out')


		
	

def main(args):
	data = ["EN", "FR"]
	weights_FR= [1.5, 1, 7, 0, 0.1] #best combination for FR dataset 
	weights_EN= [1,3,6,2,1] #best combination for EN dataset 
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