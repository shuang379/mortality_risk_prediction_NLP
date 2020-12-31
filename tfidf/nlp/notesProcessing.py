import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
import collections
import csv

def pre_process(text):

	# convert to lowercase
	text = text.lower()

	# remove punctuations and digits
	text = re.sub("(\\d)+"," ",text)
	text = ' '.join(word.strip(string.punctuation) for word in text.split())

	return text

def generate_bow(record):
	
	text = record['clean_text']

	# tokenize
	words = word_tokenize(text)

	# lemmatizing
	lemmatizer = WordNetLemmatizer()
	lemWords = [lemmatizer.lemmatize(w) for w in words]

	# remove stop words
	stop_words = set(stopwords.words("english"))
	filtered_words = [w for w in lemWords if not w in stop_words]

	# count words
	counter = collections.Counter(filtered_words)

 		with open(r'wordCount.csv', 'a', newline='') as f:
			fieldnames = ['subject_id', 'word', 'count', 'charttime']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writerow({'subject_id': record['subject_id'], 'word': element, 'count': counter[element], 'charttime': record['charttime']})

if __name__ == '__main__':

	# load notes
	df = pd.read_csv('../data/noteevents.csv', low_memory=False)
	# print(df.index[df['charttime']=='2120-02-15 11:39:00'].tolist())

	# pre process: convert to lowercase & remove punctuations/digits
	df['clean_text'] = df['text'].apply(pre_process)

	# generate_bow(df.iloc[0])

	# create csv with schema: subject_id, word, count, charttime
	for i in range(df.shape[0]):
		print('round', i)
		generate_bow(df.iloc[i]) 

	

	

	

	
	

