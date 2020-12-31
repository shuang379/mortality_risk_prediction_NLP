import pandas as pd
import numpy as np

wordCount = pd.read_csv("../data/wordCount.csv")
wordCount_grouped_id = wordCount.groupby(['subject_id', 'word']).sum().reset_index()

bag_of_word = list(set(wordCount_grouped_id['word']))
list_of_id = list(set(wordCount_grouped_id['subject_id']))

m = len(bag_of_word)
n = len(list_of_id)

term_count = pd.DataFrame([wordCount_grouped_id['subject_id'].value_counts()]).transpose().reset_index()
term_count.rename(columns={'index':'subject_id', 'subject_id':'term_count'},inplace=True)
word_count = pd.DataFrame([wordCount_grouped_id['word'].value_counts()]).transpose().reset_index()
word_count.rename(columns={'index':'word', 'word':'word_count'},inplace=True)


wordCount_grouped_id = wordCount_grouped_id.merge(term_count, on='subject_id')
wordCount_grouped_id = wordCount_grouped_id.merge(word_count, on='word')
wordCount_grouped_id['tfidf'] = wordCount_grouped_id['count'] / wordCount_grouped_id['term_count'] * np.log((n+1) / (wordCount_grouped_id['word_count'] + 1))

output_tfidf = wordCount_grouped_id[['subject_id','word','tfidf']]
output_tfidf.to_csv("../data/wordCount_tfidf.csv", index=False)
