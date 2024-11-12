import os
import pandas as pd
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import get_problem_set, get_question_id, get_problem_text, get_segment_text

def get_retrieval_threshold(method):
    if method == "jaccard_similarity":
        return constants.JACCARD_RETRIEVAL_THRESHOLD
    elif method == "bm25":
        return constants.BM25_RETRIEVAL_THRESHOLD
    elif method == "tfidf":
        return constants.TFIDF_RETRIEVAL_THRESHOLD
    else:
        return 0

def run_jaccard_similarity(segment_text, problem_text):
    segment_words = set(nltk.word_tokenize(segment_text))
    problem_words = set(nltk.word_tokenize(problem_text))

    intersection = len(segment_words.intersection(problem_words))
    union = len(segment_words.union(problem_words))
    jaccard_similarity = intersection / union if union != 0 else 0
    return jaccard_similarity

def run_bm25_similarity(segment_text, bm25):
    tokenized_segment_text = nltk.word_tokenize(segment_text)
    bm25_similarities = bm25.get_scores(tokenized_segment_text)
    return bm25_similarities

def run_tfidf_similarity(segment_text, vectorizer, X):
    vectorized_segment_text = vectorizer.transform([segment_text])
    tfidf_similarities = cosine_similarity(vectorized_segment_text, X).flatten()
    return tfidf_similarities

def retrieve(transcript_df, method, segment_key, save_score=False):
    problem_set = get_problem_set(transcript_df)
    problem_set_dir = os.path.join(constants.OCR_DIR, constants.MATH_DIR, problem_set)

    retrieval_threshold = get_retrieval_threshold(method)

    retrieved_question_ids = defaultdict(float)
    retrieved_similarity_scores = defaultdict(float)

    problem_set_corpus = [get_problem_text(os.path.join(problem_set_dir, problem)) for problem in os.listdir(problem_set_dir)
                          if os.path.splitext(problem)[1].lower() == ".txt"]
    
    tokenized_problem_set_corpus = [doc.split(" ") for doc in problem_set_corpus]
    bm25 = BM25Okapi(tokenized_problem_set_corpus)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(problem_set_corpus)

    segment_numbers = transcript_df[segment_key].unique().tolist()
    for segment_number in segment_numbers:
        segment_text = get_segment_text(transcript_df, segment_number, segment_key)

        similarity_scores = defaultdict(float)
        bm_similarities = run_bm25_similarity(segment_text, bm25)
        tfidf_similarities = run_tfidf_similarity(segment_text, vectorizer, X)

        index = 0
        for problem_filename in os.listdir(problem_set_dir):
            if os.path.splitext(problem_filename)[1].lower() != ".txt":
                continue
            problem_filepath = os.path.join(problem_set_dir, problem_filename)
            question_id = get_question_id(problem_filename)
            problem_text = get_problem_text(problem_filepath)

            # calculate similarity score between segment and each problem
            if method == "jaccard_similarity":
                similarity_score = run_jaccard_similarity(segment_text, problem_text)
                similarity_scores[question_id] = similarity_score
            elif method == "bm25":
                similarity_scores[question_id] = bm_similarities[index]
            elif method == "tfidf":
                similarity_scores[question_id] = tfidf_similarities[index]
            index += 1
        # retrieve most similar problem
        retrieved_question_id = max(similarity_scores, key=similarity_scores.get)
        if method == "bm25":
            # calculate normalized similarity score
            top_similarity_scores = sorted(similarity_scores.values(), reverse=True)[:constants.MIN_NUM_OF_PROBLEMS_IN_PROBLEM_SET]
            if sum(top_similarity_scores) > 0:
                normalized_retrieved_score = similarity_scores[retrieved_question_id] / sum(top_similarity_scores)
            else: 
                normalized_retrieved_score = 0
            
            # apply retrieval threshold
            if normalized_retrieved_score < retrieval_threshold:
                retrieved_question_ids[segment_number] = -1
            else:
                retrieved_question_ids[segment_number] = retrieved_question_id
            
            if save_score:
                retrieved_similarity_scores[segment_number] = normalized_retrieved_score

        else:
            # apply retrieval threshold
            if similarity_scores[retrieved_question_id] < retrieval_threshold:
                retrieved_question_ids[segment_number] = -1
            else:
                retrieved_question_ids[segment_number] = retrieved_question_id
            
            if save_score:
                retrieved_similarity_scores[segment_number] = similarity_scores[retrieved_question_id]

    # populate transcript df with retrieved question id's
    for row_idx, row in transcript_df.iterrows():
        segment_number = row[segment_key]
        transcript_df.loc[row_idx, constants.PRED_QUESTION_KEY] = int(retrieved_question_ids[segment_number])
        if save_score:
            transcript_df.loc[row_idx, constants.SIMILARITY_SCORE_KEY] = retrieved_similarity_scores[segment_number]
            
    return transcript_df