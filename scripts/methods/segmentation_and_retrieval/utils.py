import os
import pandas as pd
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import format_transcript, get_problem_set, get_problems

def get_prompts(transcript_df, prompt_num):
    system_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.SYSTEM_DIR, constants.SEGMENTATION_AND_RETRIEVAL_PROMPT_FILENAME.format(prompt_num=prompt_num))
    user_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.USER_DIR, constants.SEGMENTATION_AND_RETRIEVAL_PROMPT_FILENAME.format(prompt_num=prompt_num))
    transcript_str = format_transcript(transcript_df, transcript_format='line_index')
    problem_set = get_problem_set(transcript_df)
    problems_str = get_problems(problem_set)
    with open(system_prompt_fpath, 'r') as system_prompt_file:
        system_prompt_lines = system_prompt_file.readlines()
        system_prompt = ''.join(system_prompt_lines)
    with open(user_prompt_fpath, 'r') as user_prompt_file:
        user_prompt_lines = user_prompt_file.readlines()
        user_prompt = ''.join(user_prompt_lines[:-1])
        user_prompt = user_prompt.format(transcript=transcript_str, problems=problems_str)
        user_prompt = user_prompt + user_prompt_lines[-1]
    return system_prompt, user_prompt

def annotate_transcript(transcript_df, response):
    """
    response format: [{"start_line_idx": <first line index of segment 1>, "end_line_idx": <last line index of segment 1>, "problem_id": <ID of problem discussed in segment 1>}, ... {"start_line_idx": <first line index of segment n>, "end_line_idx": <last line index of segment n>, "problem_id": <ID of problem discussed in segment n>}]
    """
    # Populate PRED_SEGMENT_KEY
    boundary_idx_lst = []
    for res in response:
        start_idx, end_idx = res['start_line_index'], res['end_line_index']
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
            # If last segment, let segment end on the last row of the transcript
            end_idx = len(transcript_df) - 1 
        else:
            end_idx = boundary_idx_lst[i+1] - 1 
        transcript_df.loc[start_idx:end_idx, constants.PRED_SEGMENT_KEY] = segment_number # contrary to usual python slices, both the start and the stop are included
        segment_number += 1
    
    # Populate PRED_QUESTION_KEY
    for res in response:
        start_idx, end_idx = res['start_line_index'], res['end_line_index']
        transcript_df.loc[start_idx:end_idx, constants.PRED_QUESTION_KEY] = res['problem_id']

    return transcript_df