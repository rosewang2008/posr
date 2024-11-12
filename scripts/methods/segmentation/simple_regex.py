import os
import re
import sys
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
sys.path.append(os.getcwd())

from scripts import constants

TRAINING_TRANSCRIPT_DIR = constants.TRAIN_DIR

def get_top_k_words():
    count = {}
    tokenizer = RegexpTokenizer(r"\b\w+(?:'\w+)?\b")

    # get count of segment words in TRAINING set of transcripts
    for transcript in os.listdir(TRAINING_TRANSCRIPT_DIR):
        transcript_fpath = os.path.join(TRAINING_TRANSCRIPT_DIR, transcript)
        transcript_df = pd.read_csv(transcript_fpath)

        for row_idx, row in transcript_df.iterrows():
            if row[constants.SEGMENTATION_KEY] == 1:
                line = row[constants.TEXT_KEY].lower()
                words = tokenizer.tokenize(line)
                for word in words:
                    if word not in stopwords.words('english'):
                        if word not in count:
                            count[word] = 0
                        count[word] += 1

    return sorted(count, key=count.get, reverse=True)

def segment(transcript_df, word_bank):
  segment_number = 0

  for row_idx, row in transcript_df.iterrows():
    if row_idx == 0:
      segment_number += 1
    else:
      text = str(row[constants.TEXT_KEY]).lower()
      for word in word_bank:
        if re.search(word.lower(), text):
          segment_number += 1
          break
    transcript_df.loc[row_idx, constants.PRED_SEGMENT_KEY] = segment_number

  return transcript_df

def segment_by_transition_words(transcript_df):
  # refined list of transitionary/key words
  SEGMENT_WORDS = ['0ver', 'continue', 'example', 'first', 'next', 'practice', 'problem', 'question', 'start', 'topic']
  return segment(transcript_df, SEGMENT_WORDS)

def segment_by_top_k(transcript_df, k):
  # list of segment words based on top k words
  TOP_K_WORDS = get_top_k_words()
  SEGMENT_WORDS = TOP_K_WORDS[:k]
  return segment(transcript_df, SEGMENT_WORDS)
