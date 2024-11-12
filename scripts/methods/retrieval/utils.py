import os
import pandas as pd
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import format_transcript, get_problems

def get_prompts(transcript_df, problem_set, prompt_num, transcript_format):
    system_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.SYSTEM_DIR, constants.RETRIEVAL_PROMPT_FILENAME.format(prompt_num=prompt_num))
    user_prompt_fpath = os.path.join(constants.PROMPTS_DIR, constants.USER_DIR, constants.RETRIEVAL_PROMPT_FILENAME.format(prompt_num=prompt_num))
    transcript_str = format_transcript(transcript_df, transcript_format=transcript_format) 
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

def format_response_for_segment_retrieval(response):
    response = response.lower()
    if response == "null":
        formatted_response = None 
    else:
        try:
            formatted_response = int(response)
        except ValueError:
            print(response)
            formatted_response = None
    return formatted_response

def annotate_transcript(transcript_df, segment_key, response):
    """
    response format: [{'segment_id': <segment ID>, 'problem_id': <problem ID>}, ...]
    """
    for res in response:
        segment_id, problem_id = res['segment_id'], res['problem_id']
        transcript_df.loc[transcript_df[segment_key] == segment_id, constants.PRED_QUESTION_KEY] = problem_id 
    return transcript_df
