import os
import sys
sys.path.append(os.getcwd())

from scripts import constants
from methods.segmentation.utils import get_embeddings, get_boundary_indices
    
def annotate_transcript(transcript_df, segmentation):
    boundary_indices = get_boundary_indices(segmentation)
    segment_number = 1
    for i in range(len(boundary_indices)):
        start_idx = boundary_indices[i]
        if i == len(boundary_indices) - 1:  
            # if last segment, let segment end on the last row of the transcript
            end_idx = len(transcript_df) - 1 
        else:
            end_idx = boundary_indices[i+1] - 1 
        transcript_df.loc[start_idx:end_idx, constants.PRED_SEGMENT_KEY] = segment_number # contrary to usual python slices, both the start and the stop are included
        segment_number += 1
    return transcript_df

def segment(transcript_df, c99_model, encoder_model_name):
    embeddings = get_embeddings(transcript_df, encoder_model_name=encoder_model_name)
    segmentation = c99_model.segment(embeddings)
    transcript_df = annotate_transcript(transcript_df, segmentation)
    return transcript_df
    