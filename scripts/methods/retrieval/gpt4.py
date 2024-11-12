import os
import pandas as pd
from collections import defaultdict
import sys
sys.path.append(os.getcwd())

from scripts import constants
from scripts.methods.utils import get_problem_set, openai_completion_with_backoff, format_response_content, track_model_usage, reset_model_usage_data
from scripts.methods.retrieval.utils import get_prompts, format_response_for_segment_retrieval, annotate_transcript

def retrieve(transcript_df, segment_key, prompt_num, model='gpt-4-turbo'):
    transcript_df = reset_model_usage_data(transcript_df)
    problem_set = get_problem_set(transcript_df)
    system_prompt, user_prompt = get_prompts(transcript_df, problem_set, prompt_num, transcript_format='segment_number')
    response = openai_completion_with_backoff(
        model=model, 
        temperature=0.0, 
        messages=[
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': user_prompt}
        ]
    )
    # Annotate transcript_df with response content
    transcript_df = track_model_usage(
        model=model,
        transcript_df=transcript_df,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response
    )
    response_content = format_response_content(response.choices[0].message.content)
    transcript_df = annotate_transcript(transcript_df, segment_key, response_content)
    return transcript_df

def retrieve_segment_by_segment(transcript_df, segment_key, prompt_num, model='gpt-4-turbo'):
    transcript_df = reset_model_usage_data(transcript_df)
    problem_set = get_problem_set(transcript_df)
    for segment_id in transcript_df[segment_key].unique():
        segment_transcript_df = transcript_df[transcript_df[segment_key] == segment_id]
        system_prompt, user_prompt = get_prompts(segment_transcript_df, problem_set, prompt_num, transcript_format='speaker_only')
        response = openai_completion_with_backoff(
            model=model, 
            temperature=0.0, 
            messages=[
                {'role': 'system', 'content': system_prompt}, 
                {'role': 'user', 'content': user_prompt}
            ]
        )
        # Annotate transcript_df with response content
        transcript_df = track_model_usage(
            model=model,
            transcript_df=transcript_df,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            segment_key=segment_key,
            segment_id=segment_id            
        )
        response_content = format_response_for_segment_retrieval(response.choices[0].message.content)
        transcript_df.loc[transcript_df[segment_key] == segment_id, constants.PRED_QUESTION_KEY] = response_content  # annotate transcript
    return transcript_df