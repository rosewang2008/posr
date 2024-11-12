import os
import pandas as pd
import numpy as np
import json
import re
import sys
sys.path.append(os.getcwd())

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from scripts import constants
from openai import OpenAI
import anthropic
import transformers
import torch

def format_transcript(transcript_df, transcript_format='line_index'):
    transcript_str = ""
    if transcript_format == 'line_index':
        for idx, row in transcript_df.iterrows():
            transcript_str += f"{idx} {row['speaker']}: {row['text']}\n"
    elif transcript_format == 'segment_number':
        for idx, row in transcript_df.iterrows():
            transcript_str += f"{row[constants.ACTUAL_SEGMENT_KEY]} {row['speaker']}: {row['text']}\n"
    elif transcript_format == 'speaker_only':
        for idx, row in transcript_df.iterrows():
            transcript_str += f"{row['speaker']}: {row['text']}\n"
    return transcript_str

def get_problem_set(transcript_df):
    return str(transcript_df['problem_set'][0])[:3] 

def get_question_id(problem_filename):
    pattern = r'\d+'
    question_id = re.search(pattern, problem_filename).group()
    return question_id

def get_segment_text(transcript_df, segment_number, segment_key):
    segment_df = transcript_df[transcript_df[segment_key] == segment_number]
    segment_text = ' '.join(segment_df['text'].astype(str))
    return segment_text

def get_problem_text(problem_filepath):
    with open(problem_filepath, 'r') as file:
        problem_text = file.read()
        return problem_text

def get_problems(problem_set):
    problem_set_dir = os.path.join(constants.OCR_DIR, str(problem_set))
    problems_str = ""
    problem_fnames = [fname for fname in os.listdir(problem_set_dir) if os.path.splitext(fname)[1].lower() == ".txt"]
    problem_ids = sorted([int(get_question_id(fname)) for fname in problem_fnames])
    for problem_id in problem_ids:
        problem_fname = f'problem{problem_id}.txt'
        problem_fpath = os.path.join(problem_set_dir, problem_fname)
        problem_text = get_problem_text(problem_fpath)
        problems_str += f'Problem ID {problem_id}:\n{problem_text}\n'
    return problems_str

def get_claude_model(claude_method):
    if "haiku" in claude_method:
        return constants.CLAUDE_HAIKU
    elif "sonnet" in claude_method:
        return constants.CLAUDE_SONNET
    elif "opus" in claude_method:
        return constants.CLAUDE_OPUS

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def openai_completion_with_backoff(**kwargs):
    client = OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def anthropic_call_with_backoff(**kwargs):
    client = anthropic.Anthropic()
    response = client.messages.create(**kwargs)
    return response

def format_response_content(content, target_start="[", target_end="]"):
    content = content.replace(" ", "").replace("\n", "")  # address spacing errors in GPT-4's output
    if content.startswith(target_start) and content.endswith(target_end):
        target_content = json.loads(content)
    else:
        start_idx = content.find(target_start)
        end_idx = content.rfind(target_end)
        # target_content = json.loads(content[start_idx:end_idx+len(target_end)])
        json_string = content[start_idx:end_idx+len(target_end)]
        try: 
            target_content = json.loads(json_string)
        except json.JSONDecodeError as e:
            target_content = []
            print(f"Error decoding JSON: {e}")
            print(f"JSON string: {json_string}")
    return target_content

def track_model_usage(transcript_df, system_prompt, user_prompt, response, model, segment_key=None, segment_id=None):
    # Determine the row index to annotate based on segment_key and segment_id
    if segment_key is not None and segment_id is not None:
        row_idx = transcript_df.loc[transcript_df[segment_key] == segment_id].index[0]
    else:
        row_idx = 0  # Default to annotating the first row if segment_key and segment_id are not provided

    # Annotate transcript with model usage in the first row
    transcript_df.loc[row_idx, constants.SYSTEM_PROMPT_KEY] = system_prompt
    transcript_df.loc[row_idx, constants.USER_PROMPT_KEY] = user_prompt
    if "gpt" in model:
        transcript_df.loc[row_idx, constants.RAW_OUTPUT_KEY] = response.choices[0].message.content
        transcript_df.loc[row_idx, constants.NUM_PROMPT_TOKENS_KEY] = response.usage.prompt_tokens
        transcript_df.loc[row_idx, constants.NUM_COMPLETION_TOKENS_KEY] = response.usage.completion_tokens
        transcript_df.loc[row_idx, constants.NUM_TOTAL_TOKENS_KEY] = response.usage.total_tokens
    elif "claude" in model:
        transcript_df.loc[row_idx,constants.RAW_OUTPUT_KEY] = response.content[0].text
        transcript_df.loc[row_idx, constants.NUM_PROMPT_TOKENS_KEY] = response.usage.input_tokens
        transcript_df.loc[row_idx, constants.NUM_COMPLETION_TOKENS_KEY] = response.usage.output_tokens
        transcript_df.loc[row_idx, constants.NUM_TOTAL_TOKENS_KEY] = response.usage.input_tokens + response.usage.output_tokens
    return transcript_df

def reset_model_usage_data(transcript_df):
    columns_to_clear = [
        constants.SYSTEM_PROMPT_KEY,
        constants.USER_PROMPT_KEY,
        constants.RAW_OUTPUT_KEY,
        constants.NUM_PROMPT_TOKENS_KEY,
        constants.NUM_COMPLETION_TOKENS_KEY,
        constants.NUM_TOTAL_TOKENS_KEY
    ]
    transcript_df[columns_to_clear] = None
    return transcript_df