import os
import pandas as pd
import json
import re
import sys
sys.path.append(os.getcwd())

from scripts.methods.utils import get_claude_model, anthropic_call_with_backoff, format_response_content, track_model_usage 
from scripts.methods.segmentation_and_retrieval.utils import get_prompts, annotate_transcript

def segment_and_retrieve(transcript_df, prompt_num, method):
    system_prompt, user_prompt = get_prompts(transcript_df, prompt_num)
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
    transcript_df = annotate_transcript(transcript_df, response_content)
    return transcript_df