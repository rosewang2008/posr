import os
import pandas as pd
from tqdm import tqdm

from constants import (
    UNFORMATTED_TRANSCRIPT_DATA_DIR,
    FORMATTED_TRANSCRIPT_DATA_DIR, 
    SEGMENTATION_KEY,
    RETRIEVAL_KEY, 
    PRED_SEGMENT_KEY,
    ACTUAL_SEGMENT_KEY,
    PRED_QUESTION_KEY,
    ACTUAL_QUESTION_KEY
)

def format_annotated_transcripts():
    for transcript_filename in tqdm(os.listdir(UNFORMATTED_TRANSCRIPT_DATA_DIR)):
        transcript_path = os.path.join(UNFORMATTED_TRANSCRIPT_DATA_DIR, transcript_filename)
        if os.path.splitext(transcript_path)[1].lower() != ".csv":
            continue
        transcript_df = pd.read_csv(transcript_path)

        output_dir = FORMATTED_TRANSCRIPT_DATA_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(FORMATTED_TRANSCRIPT_DATA_DIR, transcript_filename)

        transcript_df[ACTUAL_SEGMENT_KEY] = None
        transcript_df[PRED_SEGMENT_KEY] = None
        transcript_df[ACTUAL_QUESTION_KEY] = None
        transcript_df[PRED_QUESTION_KEY] = None

        segment_number = 0
        question_id = None

        for row_idx, row in transcript_df.iterrows():
            # check if it's the first row of a segment
            if row[SEGMENTATION_KEY] == 1:
                # get question_id for the current segment
                question_id = row[RETRIEVAL_KEY]
                # increment segment index
                segment_number += 1
            
            # populate the ACTUAL_SEGMENT_KEY and ACTUAL_QUESTION_KEY columns
            transcript_df.at[row_idx, ACTUAL_SEGMENT_KEY] = segment_number 
            transcript_df.at[row_idx, ACTUAL_QUESTION_KEY] = question_id

        transcript_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    format_annotated_transcripts()