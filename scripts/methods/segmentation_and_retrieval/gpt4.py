import os
import pandas as pd
import json
import sys
sys.path.append(os.getcwd())

from scripts.methods.utils import openai_completion_with_backoff, format_response_content, track_model_usage
from scripts.methods.segmentation_and_retrieval.utils import get_prompts, annotate_transcript

def segment_and_retrieve(transcript_df, prompt_num):
    system_prompt, user_prompt = get_prompts(transcript_df, prompt_num)
    model = 'gpt-4-turbo'
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
    transcript_df = annotate_transcript(transcript_df, response_content)
    return transcript_df