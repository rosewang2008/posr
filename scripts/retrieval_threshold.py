import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.append(os.getcwd())

from scripts import constants
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from methods.retrieval import ngrams
from methods.retrieval.colbert_retrieval import colbert_retrieval
from metrics import get_transitions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str)
args = parser.parse_args()

THRESHOLDS = {
    'jaccard_similarity': {
        'start': 0,
        'stop': 0.5,
        'step': 0.01
    },
    'tfidf': {
        'start': 0,
        'stop': 0.7,
        'step': 0.01
    },
    'bm25': {
        'start': 0,
        'stop': 1, 
        'step': 0.01 
    },
    'colbert_retrieval': {
        'start': 0,
        'stop': 1,
        'step': 0.01
    }
}

def get_retrieval_score(df):
    ground_truth_questions = df[constants.ACTUAL_QUESTION_KEY].to_numpy()
    predicted_questions = df[constants.PRED_QUESTION_KEY].to_numpy()
    is_correct = ground_truth_questions == predicted_questions
    accuracy = np.mean(is_correct)
    return accuracy

def run_retrieval(args):
    results_dir = os.path.join(constants.RETRIEVAL_THRESHOLD_DIR, args.method, "transcripts")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    transcript_dir = constants.TRAIN_DIR
    transcript_fnames = os.listdir(transcript_dir)

    for transcript_fname in tqdm(transcript_fnames):
        transcript_fpath = os.path.join(transcript_dir, transcript_fname)
        transcript_df = pd.read_csv(transcript_fpath)

        result_fpath = os.path.join(results_dir, transcript_fname)
        if os.path.exists(result_fpath):
            continue

        if args.method == 'jaccard_similarity' or args.method == 'tfidf' or args.method == 'bm25':
            transcript_df = ngrams.retrieve(transcript_df, args.method, constants.ACTUAL_SEGMENT_KEY, save_score=True)
        elif args.method == 'colbert_retrieval':
            transcript_df = colbert_retrieval.retrieve(transcript_df, constants.ACTUAL_SEGMENT_KEY, save_score=True)

        transition_indices = get_transitions(transcript_df, constants.ACTUAL_SEGMENT_KEY, as_indices=True, include_first=True)
        score_df = transcript_df.loc[transition_indices, [constants.ACTUAL_SEGMENT_KEY, constants.ACTUAL_QUESTION_KEY, constants.PRED_QUESTION_KEY, constants.SIMILARITY_SCORE_KEY]]
        score_df.to_csv(result_fpath, index=False)

    test_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in tqdm(enumerate(kf.split(transcript_fnames))):
        train_fnames = [transcript_fnames[i] for i in train_idx]
        test_fnames = [transcript_fnames[i] for i in test_idx]

        train_scores = defaultdict(float)
        start, stop, step = THRESHOLDS[args.method]['start'], THRESHOLDS[args.method]['stop'], THRESHOLDS[args.method]['step']
        for threshold in np.arange(start, stop, step):
            scores = []
            for train_fname in tqdm(train_fnames):
                train_fpath = os.path.join(results_dir, train_fname)
                df = pd.read_csv(train_fpath)
                df[constants.ACTUAL_QUESTION_KEY] = df[constants.ACTUAL_QUESTION_KEY].fillna(-1)
                df.loc[df[constants.SIMILARITY_SCORE_KEY] < threshold, constants.PRED_QUESTION_KEY] = -1
                score = get_retrieval_score(df)
                scores.append(score)
            train_scores[threshold] = np.mean(scores)

        plt.plot(train_scores.keys(), train_scores.values())
        plt.xlabel('Threshold')
        plt.ylabel('Retrieval Accuracy')
        plt.title(f'{args.method} Retrieval Accuracy vs. Threshold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(constants.RETRIEVAL_THRESHOLD_DIR, args.method, f'fold{fold}.png'))

        best_train_threshold = max(train_scores, key=train_scores.get)
        scores = []
        for test_fname in test_fnames:
            test_fpath = os.path.join(results_dir, test_fname)
            df = pd.read_csv(test_fpath)
            df[constants.ACTUAL_QUESTION_KEY] = df[constants.ACTUAL_QUESTION_KEY].fillna(-1)
            df.loc[df[constants.SIMILARITY_SCORE_KEY] < best_train_threshold, constants.PRED_QUESTION_KEY] = -1
            score = get_retrieval_score(df)
            scores.append(score)
        test_scores.append((best_train_threshold, np.mean(scores)))
        print(f"fold {fold}")
        print(f"threshold: {best_train_threshold}")
        print(f"accuracy: {np.mean(scores)}")

    with open(os.path.join(constants.RETRIEVAL_THRESHOLD_DIR, args.method, "test_scores.txt"), "w") as f:
        for fold, (threshold, score) in enumerate(test_scores):
            f.write(f"fold {fold}\n")
            f.write(f"threshold: {threshold}\n")
            f.write(f"accuracy: {score}\n\n")

if __name__ == "__main__":
    assert args.method in ['jaccard_similarity', 'tfidf', 'bm25', 'colbert_retrieval'], "Invalid method"
    run_retrieval(args)
