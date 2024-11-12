from nltk.tokenize import TextTilingTokenizer

def segment(transcript_df, text_column, output_column): 
    tt = TextTilingTokenizer()
    text = '\n\n'.join([f"{row['speaker']}: {row[text_column]}" for _, row in transcript_df.iterrows()])
    # s, ss, d, boundaries = tt.tokenize(text)
    segmented_text = tt.tokenize(text)

    # Add segment
    segment_id = 0
    segmented_text_idx = 0
    transcript_df[output_column] = None
    for i, row in transcript_df.iterrows():
        # Get first sentence of the segment. Find where this sentence is in the segmented_text
        seg_sentence = segmented_text[segmented_text_idx].strip().split('\n\n')[0].strip()
        # Index of ":" in the sentence
        seg_sentence = seg_sentence[seg_sentence.index(':')+1:].strip()

        # Find the start of the segment in the transcript
        if row[text_column] == seg_sentence:
            segment_id += 1
            segmented_text_idx = min(segmented_text_idx + 1, len(segmented_text) - 1)
        transcript_df.loc[i, output_column] = segment_id
    return transcript_df