import os
import pandas as pd
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import format_transcript
import torch
from sentence_transformers import SentenceTransformer

def get_prompts(transcript_df, prompt_num):
    system_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.SYSTEM_DIR, constants.SEGMENTATION_PROMPT_FILENAME.format(prompt_num=prompt_num))
    user_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.USER_DIR, constants.SEGMENTATION_PROMPT_FILENAME.format(prompt_num=prompt_num))
    transcript_str = format_transcript(transcript_df, transcript_format='line_index')
    with open(system_prompt_fpath, 'r') as system_prompt_file:
        system_prompt_lines = system_prompt_file.readlines()
        system_prompt = ''.join(system_prompt_lines)
    with open(user_prompt_fpath, 'r') as user_prompt_file:
        user_prompt_lines = user_prompt_file.readlines()
        user_prompt = ''.join(user_prompt_lines)
        user_prompt = user_prompt.format(transcript=transcript_str)
    return system_prompt, user_prompt

def annotate_transcript(transcript_df, response):
    """
    response format: [[<first line index of segment 1>, <last line index of segment 1>], ..., [<first line index of segment n>, <last line index of segment n>]]
    """
    boundary_idx_lst = []
    for sublist in response:
        start_idx, end_idx = sublist[0], sublist[1]
        boundary_idx_lst.append(start_idx)
        end_idx = end_idx + 1 if end_idx < (len(transcript_df) - 1) else end_idx
        boundary_idx_lst.append(end_idx)
    if 0 not in boundary_idx_lst:
        boundary_idx_lst.insert(0, 0)
    boundary_idx_lst = sorted(list(set(boundary_idx_lst)))
    
    segment_number = 1
    for i in range(len(boundary_idx_lst)):
        start_idx = boundary_idx_lst[i]
        if i == len(boundary_idx_lst) - 1:  
            # if last segment, let segment end on the last row of the transcript
            end_idx = len(transcript_df) - 1 
        else:
            end_idx = boundary_idx_lst[i+1] - 1 
        transcript_df.loc[start_idx:end_idx, constants.PRED_SEGMENT_KEY] = segment_number # contrary to usual python slices, both the start and the stop are included
        segment_number += 1
    return transcript_df

def get_transcript_lines(transcript_df):
    lines = []
    for idx, row in transcript_df.iterrows():
        line = f"{row['speaker']}: {row['text']}"
        lines.append(line)
    return lines

def get_embeddings(transcript_df, encoder_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer(encoder_model_name)
    embedder.to(device)
    lines = get_transcript_lines(transcript_df)
    with torch.no_grad():
        embeddings = embedder.encode(lines)
    return embeddings

def get_boundary_indices(segmentation):
    """
    segmentation: list of 1s and 0s
    """
    return [i for i, val in enumerate(segmentation) if val == 1]