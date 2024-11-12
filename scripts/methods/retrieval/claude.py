import os
import pandas as pd
import sys
sys.path.append(os.getcwd())

from scripts.methods.utils import get_problem_set, get_claude_model, anthropic_call_with_backoff, format_response_content, track_model_usage, reset_model_usage_data
from scripts.methods.retrieval.utils import get_prompts, format_response_for_segment_retrieval, annotate_transcript
from scripts import constants

def retrieve(transcript_df, segment_key, prompt_num, method):
    transcript_df = reset_model_usage_data(transcript_df)
    problem_set = get_problem_set(transcript_df)
    system_prompt, user_prompt = get_prompts(transcript_df, problem_set, prompt_num, transcript_format='segment_number')
    model = get_claude_model(method)
    response = anthropic_call_with_backoff(
        model=model,
        max_tokens=1000,
        temperature=0.0,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}]
    )
    # Annotate transcript_df with response content
    transcript_df = track_model_usage(
        model=model,
        transcript_df=transcript_df,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response
    )
    response_content = format_response_content(response.content[0].text)
    transcript_df = annotate_transcript(transcript_df, segment_key, response_content)
    return transcript_df

def retrieve_segment_by_segment(transcript_df, segment_key, prompt_num, method):
    transcript_df = reset_model_usage_data(transcript_df)
    model = get_claude_model(method)
    problem_set = get_problem_set(transcript_df)
    for segment_id in transcript_df[segment_key].unique():
        segment_transcript_df = transcript_df[transcript_df[segment_key] == segment_id]
        system_prompt, user_prompt = get_prompts(segment_transcript_df, problem_set, prompt_num, transcript_format='speaker_only')
        response = anthropic_call_with_backoff(
            model=model,
            max_tokens=1000,
            temperature=0.0,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}]
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
        response_content = format_response_for_segment_retrieval(response.content[0].text)
        transcript_df.loc[transcript_df[segment_key] == segment_id, constants.PRED_QUESTION_KEY] = response_content  # annotate transcript
    return transcript_df