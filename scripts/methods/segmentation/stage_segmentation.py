from hmmlearn import hmm
import sys
import os
sys.path.append(os.getcwd())
import pickle
import pandas as pd
import numpy as np
from scripts import constants
from methods.segmentation.utils import get_embeddings

def get_embeddings_for_directory(encoder_name, transcript_directory, embedding_save_path):
    # Go through all train transcripts and get embeddings
    if os.path.exists(embedding_save_path):
        with open(embedding_save_path, 'rb') as f:
            train_embeddings = pickle.load(f) # i = transcript, j = length of transcript

    else:
        train_embeddings = []
        for f in os.listdir(transcript_directory):
            transcript_fpath = os.path.join(transcript_directory, f)
            transcript_df = pd.read_csv(transcript_fpath)
            embeddings = get_embeddings(transcript_df, encoder_name)
            train_embeddings.append(embeddings)
        with open(embedding_save_path, 'wb') as f:
            pickle.dump(train_embeddings, f)
    return train_embeddings

def get_X_length_from_embeddings(embeddings):
    embed_X = []
    length = []
    for i in range(0, len(embeddings)):
        length.append(len(embeddings[i]))
        for j in range(0, len(embeddings[i])):
            embed_X.append(np.array(embeddings[i][j]))
    return embed_X, length


def get_train_hmm_data(encoder_name):
    # TRAIN_EMBEDDINGS_FPATH = "scripts/methods/segmentation/{encoder_name}_train_embeddings.pkl"
    train_embeddings = get_embeddings_for_directory(
        encoder_name,
        transcript_directory=constants.TRAIN_DIR,
        embedding_save_path=constants.TRAIN_EMBEDDINGS_FPATH.format(encoder_name=encoder_name)
    )
    embed_X, length = get_X_length_from_embeddings(train_embeddings)
    return embed_X, length

def load_and_fit_hmm_model(
        encoder_name,
        num_components=constants.AVG_NUM_SEGMENTS_PER_TRANSCRIPT,
        num_iter=constants.HMM_NUM_ITER,
        covariance_type = 'diag',
        verbose = True,
        init_params="cm",
        params="cmts"):
    np.random.seed(42)
    model = hmm.GaussianHMM(
        n_components=num_components,
        covariance_type=covariance_type,
        n_iter=num_iter,
        verbose=verbose,
        init_params=init_params,
        params=params
    )
    # Uniformly distribute probabilities across NUM_SEGMENTS
    model.startprob_ = np.full(num_components, 1/num_components)
    model.transmat_ = np.full((num_components, num_components), 1/num_components)
    embed_X, length = get_train_hmm_data(encoder_name)
    model.fit(embed_X, length)
    return model

def get_hmm_labels(embeddings, hmm_model):
    labels = []
    for embedding in embeddings:
        result = hmm_model.decode(np.array([embedding]))
        label = result[1].item()
        labels.append(label)
    return labels

def annotate_transcript_df(transcript_df, hmm_labels):
    segment_number = 1
    current_hmm_label = None
    for (i, row), hmm_label in zip(transcript_df.iterrows(), hmm_labels):
        if current_hmm_label is None:
            current_hmm_label = hmm_label
        elif current_hmm_label != hmm_label:
            segment_number += 1
            current_hmm_label = hmm_label
        transcript_df.loc[i, constants.PRED_SEGMENT_KEY] = segment_number
    return transcript_df

def segment(transcript_df, hmm_model, encoder_name):
    # num_transcript_lines x embedding_dim
    embeddings = get_embeddings(transcript_df, encoder_name)
    labels = get_hmm_labels(embeddings, hmm_model=hmm_model)
    transcript_df = annotate_transcript_df(transcript_df=transcript_df, hmm_labels=labels)
    return transcript_df
