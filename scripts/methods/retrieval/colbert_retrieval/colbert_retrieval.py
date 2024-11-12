import os
import pandas as pd
from collections import defaultdict
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import get_problem_set, get_segment_text
from ragatouille import RAGPretrainedModel

def retrieve(transcript_df, segment_key, save_score=False):
    problem_set = get_problem_set(transcript_df)
    retrieval_threshold = constants.COLBERT_RETRIEVAL_THRESHOLD
    retrieved_question_ids = defaultdict(float)
    retrieved_similarity_scores = defaultdict(float)
    
    index_path = os.path.join(constants.COLBERT_INDEXES_DIR, problem_set)
    RAG = RAGPretrainedModel.from_index(index_path)

    segment_numbers = transcript_df[segment_key].unique().tolist()
    for segment_number in segment_numbers:
        segment_text = get_segment_text(transcript_df, segment_number, segment_key)
        results = RAG.search(segment_text)
    
        # calculate normalized similarity score
        top_similarity_scores = [item['score'] for item in results[:constants.MIN_NUM_OF_PROBLEMS_IN_PROBLEM_SET]]
        if sum(top_similarity_scores) > 0:
            normalized_retrieved_score = results[0]['score'] / sum(top_similarity_scores)
        else:
            normalized_retrieved_score = 0
        
        # apply retrieval threshold
        if normalized_retrieved_score < retrieval_threshold:
            retrieved_question_ids[segment_number] = -1
        else:
            retrieved_question_ids[segment_number] = results[0]['document_id']

        if save_score:
            retrieved_similarity_scores[segment_number] = normalized_retrieved_score

    # populate transcript df with retrieved question id's
    for row_idx, row in transcript_df.iterrows():
        segment_number = row[segment_key]
        transcript_df.loc[row_idx, constants.PRED_QUESTION_KEY] = int(retrieved_question_ids[segment_number])
        if save_score:
            transcript_df.loc[row_idx, constants.SIMILARITY_SCORE_KEY] = retrieved_similarity_scores[segment_number]

    return transcript_df
