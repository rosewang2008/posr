import pandas as pd
import numpy as np
import os
import datetime

from nltk.metrics.segmentation import (
    windowdiff,
    pk
)

from scripts import constants

def calculate_model_usage_metrics(df, method, task, transcript_fname):
    if task == constants.SEGMENTATION_AND_RETRIEVAL_TASK and '->' in method:
        segmentation_method = method.split('->')[0]
        retrieval_method = method.split('->')[1]

        # Initialize number and cost of tokens
        num_input_tokens = 0
        num_output_tokens = 0
        num_total_tokens = 0
        input_cost = 0
        output_cost = 0
        
        # Calculate cost of tokens from segmentation step, if applicable
        if 'gpt-4' in segmentation_method or 'claude' in segmentation_method:
            # While the retrieval usage metrics may be stored in the df passed in, the segmentation usage metrics are not and will instead be drawn from the segmentation results files, which were generated before the retrieval step
            seg_transcript_fpath = os.path.join(constants.RESULTS_DIR, constants.SEGMENTATION_DIR, segmentation_method, "transcripts", transcript_fname)
            seg_transcript_df = pd.read_csv(seg_transcript_fpath)
            seg_usage_rows = seg_transcript_df[["num_prompt_tokens", "num_completion_tokens", "num_total_tokens"]].dropna()

            seg_num_input_tokens = seg_usage_rows["num_prompt_tokens"].sum()
            seg_num_output_tokens = seg_usage_rows["num_completion_tokens"].sum()
            seg_num_total_tokens = seg_usage_rows["num_total_tokens"].sum()

            num_input_tokens += seg_num_input_tokens
            num_output_tokens += seg_num_output_tokens
            num_total_tokens += seg_num_total_tokens

            seg_input_cost_per_1m = constants.MODEL_2_INPUT_TOKEN_COST_1M[segmentation_method]
            input_cost += seg_num_input_tokens * seg_input_cost_per_1m / 1_000_000
            seg_output_cost_per_1m = constants.MODEL_2_OUTPUT_TOKEN_COST_1M[segmentation_method]
            output_cost += seg_num_output_tokens * seg_output_cost_per_1m / 1_000_000
            
        # Calculate cost of tokens from retrieval step, if applicable
        if 'gpt-4' in retrieval_method or 'claude' in retrieval_method:
            ret_usage_rows = df[["num_prompt_tokens", "num_completion_tokens", "num_total_tokens"]].dropna()
            ret_num_input_tokens = ret_usage_rows["num_prompt_tokens"].sum()
            ret_num_output_tokens = ret_usage_rows["num_completion_tokens"].sum()
            ret_num_total_tokens = ret_usage_rows["num_total_tokens"].sum()

            num_input_tokens += ret_num_input_tokens
            num_output_tokens += ret_num_output_tokens
            num_total_tokens += ret_num_total_tokens

            ret_input_cost_per_1m = constants.MODEL_2_INPUT_TOKEN_COST_1M[retrieval_method]
            input_cost += ret_num_input_tokens * ret_input_cost_per_1m / 1_000_000
            ret_output_cost_per_1m = constants.MODEL_2_OUTPUT_TOKEN_COST_1M[retrieval_method]
            output_cost += ret_num_output_tokens * ret_output_cost_per_1m / 1_000_000
    else:
        usage_rows = df[["num_prompt_tokens", "num_completion_tokens", "num_total_tokens"]].dropna()
        num_input_tokens = usage_rows["num_prompt_tokens"].sum()
        num_output_tokens = usage_rows["num_completion_tokens"].sum()
        num_total_tokens = usage_rows["num_total_tokens"].sum()

        model = method
        input_cost_per_1m = constants.MODEL_2_INPUT_TOKEN_COST_1M[model]
        input_cost = num_input_tokens * input_cost_per_1m / 1_000_000
        output_cost_per_1m = constants.MODEL_2_OUTPUT_TOKEN_COST_1M[model]
        output_cost = num_output_tokens * output_cost_per_1m / 1_000_000

    return {
        "model_usage/num_input_tokens": num_input_tokens,
        "model_usage/num_output_tokens": num_output_tokens,
        "model_usage/num_total_tokens": num_total_tokens,
        "model_usage/input_cost": input_cost,
        "model_usage/output_cost": output_cost,
        "model_usage/total_cost": input_cost + output_cost,
        "model_usage/total_cost_per_100_transcripts": (input_cost + output_cost) * 100
    }

def calculate_iou(pred_df, true_df):
    intersection = len(pd.merge(pred_df, true_df, how='inner'))
    union = len(pred_df) + len(true_df) - intersection
    return intersection / union if union > 0 else 0

def get_srs(df, ground_truth_question_column, predicted_question_column):
    num_correct_predictions = len(df[df[predicted_question_column] == df[ground_truth_question_column]])
    total_rows = len(df)
    srs = num_correct_predictions / total_rows if total_rows > 0 else 0
    return srs

def get_time_srs(df, ground_truth_question_column, predicted_question_column):
    df[constants.END_TIME] = pd.to_datetime(df[constants.END_TIME])
    df[constants.START_TIME] = pd.to_datetime(df[constants.START_TIME])
    df['duration'] = (df[constants.END_TIME] - df[constants.START_TIME]).dt.total_seconds()
    df['is_correct'] = (df[constants.PRED_QUESTION_KEY] == df[constants.ACTUAL_QUESTION_KEY]).astype(int)
    time_srs = (df['duration'] * df['is_correct']).sum() / df['duration'].sum()
    return time_srs

def calculate_segmentation_retrieval_scores(df, ground_truth_question_column, predicted_question_column):
    srs = get_srs(df, ground_truth_question_column, predicted_question_column)
    time_srs = get_time_srs(df, ground_truth_question_column, predicted_question_column)
    return {
        "segmentation_retrieval/srs": srs,
        "segmentation_retrieval/time_srs": time_srs,
    }
  
def get_transitions(df, transition_column, as_indices=False, include_first=False):
    """
    Assumes that the transitions happen when the value of the transition_column changes

    Either gives the indices of the transitions or the values of the transitions
    """
    transitions = []
    for i, row in df.iterrows():
        if i == 0:
            if include_first:
                transitions.append(i)
            continue
        if row[transition_column] != df.iloc[i-1][transition_column]:
            if as_indices:
                transitions.append(i)
            else:
                transitions.append(row[transition_column])
    return transitions

def get_segments(transition_indices, df_len):
    """
    Takes in indices of transitions and len(df)
    
    Outputs a list of tuples of indices that each segment spans over: [<tuple of indices that segment 1 spans over>, ..., <tuple of indices that segment n spans over>]
    """
    segment_boundaries = transition_indices
    if 0 not in segment_boundaries:
        segment_boundaries.insert(0, 0)  # to make sure the output includes the first segment
    if df_len not in segment_boundaries:
        segment_boundaries.append(df_len)  # to make sure the output includes the last segment
    return [tuple(range(segment_boundaries[i], segment_boundaries[i+1])) for i in range(len(segment_boundaries)-1)]

def get_segmentation_sequence(transition_indices, seq_len):
    """
    Takes in indicies of transitions and length of sequence, i.e. len(df)
    
    Outputs a string of "0"s and "1"s where "1" represents a transition
    """
    sequence = ["0"] * seq_len
    for index in transition_indices:
        if 0 <= index < seq_len:
            sequence[index] = "1"
    return ''.join(sequence)

def get_k(df, ground_truth_column, unit, boundary="1"):
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    if unit == "line":
        ground_truth_sequence = get_segmentation_sequence(ground_truth_transitions, len(df))
        return int(round(len(ground_truth_sequence) / (ground_truth_sequence.count(boundary) * 2.0)))
    elif unit == "time":
        # 2023-03-02T15:57:30.480-08:00 from constants.startTimestamp, constants.endTimestamp
        # Compute the average time between transitions: take the difference between the start and end time of each segment
        segment_times = []
        for i in range(len(ground_truth_transitions)-1):
            start_time = df.iloc[ground_truth_transitions[i]][constants.START_TIME]
            end_time = df.iloc[ground_truth_transitions[i+1]][constants.END_TIME]

            # Convert to seconds
            start_time = pd.Timestamp(start_time).timestamp()
            end_time = pd.Timestamp(end_time).timestamp()

            segment_times.append(end_time - start_time)
        return int(round(sum(segment_times) / len(segment_times)))

def time_windowdiff(df, ground_truth_binary_column, predicted_binary_column, time_k, line_k, weighted):
    wd = 0
    for i, row in df.iterrows():
        if i == len(df) - line_k + 1:
            break
        # For each row, consider the rows within pd.Timestamp(start_time).timestamp() + time_k
        relevant_rows = df[
            # Past the current row
            (df[constants.START_TIME] >= row[constants.START_TIME]) & 
            # Within the time window
            (df[constants.START_TIME] <= row[constants.START_TIME] + datetime.timedelta(seconds=time_k))
            ]
        ground_truth = relevant_rows[ground_truth_binary_column].values
        predicted = relevant_rows[predicted_binary_column].values
        boundary_diff = abs(sum(ground_truth) - sum(predicted))

        if weighted:
            wd += boundary_diff
        else:
            wd += int(boundary_diff > 0)
    return wd / (len(df) - line_k + 1)

def time_pk(df, ground_truth_binary_column, predicted_binary_column, time_k, line_k):
    pk_value = 0
    for i, row in df.iterrows():
        if i == len(df) - line_k + 1:
            break
        # For each row, consider the rows within pd.Timestamp(start_time).timestamp() + time_k
        relevant_rows = df[
            # Past the current row
            (df[constants.START_TIME] >= row[constants.START_TIME]) & 
            # Within the time window
            (df[constants.START_TIME] <= row[constants.START_TIME] + datetime.timedelta(seconds=time_k))
            ]
        ground_truth = sum(relevant_rows[ground_truth_binary_column].values) > 0
        predicted = sum(relevant_rows[predicted_binary_column].values) > 0

        if ground_truth != predicted:
            pk_value += 1

    return pk_value / (len(df) - line_k + 1)


def get_transition_variables(ground_truth_transition_indices, predicted_transition_indices):
    """Outputs a dictionary with: 

    N_T = number of actual transitions
    N_C = number of correctly detected transitions 
    N_I = number of falsely inserted transitions
    N_M = number of missed transitions
    """
    ground_truth_transition_indices_set = set(ground_truth_transition_indices)
    predicted_transition_indices_set = set(predicted_transition_indices)
    N_T = len(ground_truth_transition_indices)
    N_C = len(ground_truth_transition_indices_set.intersection(predicted_transition_indices_set))
    N_I = len(predicted_transition_indices_set.difference(ground_truth_transition_indices_set))
    N_M = len(ground_truth_transition_indices_set.difference(predicted_transition_indices_set))

    if N_T != N_C + N_M:
        import pdb; pdb.set_trace()

    return {
        "N_T": N_T,
        "N_C": N_C,
        "N_I": N_I,
        "N_M": N_M
    }


def get_transition_recall(df, ground_truth_column, predicted_column):
    """Computes: 

    Recall = N_C / (N_C + N_M)
    """
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    predicted_transitions = get_transitions(df, predicted_column, as_indices=True)
    transition_variables = get_transition_variables(ground_truth_transitions, predicted_transitions)
    return transition_variables["N_C"] / (transition_variables["N_C"] + transition_variables["N_M"])


def get_transition_precision(df, ground_truth_column, predicted_column):
    """Computes: 

    Precision = N_C / (N_C + N_I)
    """
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    predicted_transitions = get_transitions(df, predicted_column, as_indices=True)
    transition_variables = get_transition_variables(ground_truth_transitions, predicted_transitions)
    return transition_variables["N_C"] / (transition_variables["N_C"] + transition_variables["N_I"])

def get_transition_accuracy(df, ground_truth_column, predicted_column):
    """Computes: 

    Accuracy = N_C / N_T 
    """
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    predicted_transitions = get_transitions(df, predicted_column, as_indices=True)
    transition_variables = get_transition_variables(ground_truth_transitions, predicted_transitions)
    return transition_variables["N_C"] / transition_variables["N_T"]

def get_transition_error_rate(df, ground_truth_column, predicted_column):
    """Computes: 

    Error Rate = (N_I + N_M) / (N_T + N_I)
    """
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    predicted_transitions = get_transitions(df, predicted_column, as_indices=True)
    transition_variables = get_transition_variables(ground_truth_transitions, predicted_transitions)
    return (transition_variables["N_I"] + transition_variables["N_M"]) / (transition_variables["N_T"] + transition_variables["N_I"])

def get_segmentation_iou(df, ground_truth_column, predicted_column):
    """Computes: 

    IoU = number of overlapping sentences / number of total sentences
    """
    num_overlapping = len(df[df[ground_truth_column] == df[predicted_column]])
    num_total_sentences = len(df)
    return num_overlapping / num_total_sentences

def get_greedy_iou(ground_truth_segments, predicted_segments):
    def get_iou(segment1, segment2):
        intersection = len(set(segment1).intersection(set(segment2)))
        union = len(set(segment1).union(set(segment2)))
        return intersection / union if union > 0 else 0

    matched_segments = []
    used_predicted_segments = set()
    num_unmatched_segments = 0

    for truth_segment in ground_truth_segments:
        max_iou = 0
        matched_predicted = None
        for predicted_segment in predicted_segments:
            if predicted_segment in used_predicted_segments:
                continue
            iou = get_iou(truth_segment, predicted_segment)
            if iou > max_iou:
                max_iou = iou
                matched_predicted = predicted_segment
        if matched_predicted is None:
            num_unmatched_segments += 1
        else:
            matched_segments.append((truth_segment, matched_predicted))
            used_predicted_segments.add(matched_predicted)

    iou_values = [0] * num_unmatched_segments
    for truth_segment, matched_predicted in matched_segments:
        iou = get_iou(truth_segment, matched_predicted)
        iou_values.append(iou)

    average_iou = sum(iou_values) / len(iou_values) if len(iou_values) > 0 else 0
    return average_iou

def calculate_segmentation_scores(df, ground_truth_column, predicted_column):
    ground_truth_transitions = get_transitions(df, ground_truth_column, as_indices=True)
    predicted_transitions = get_transitions(df, predicted_column, as_indices=True)
    transition_variables = get_transition_variables(ground_truth_transitions, predicted_transitions)
    # Prepend "segmentation/" to the keys
    transition_variables = {f"segmentation/{k}": v for k, v in transition_variables.items()}

    # recall = get_transition_recall(df, ground_truth_column, predicted_column)
    # precision = get_transition_precision(df, ground_truth_column, predicted_column)
    # accuracy = get_transition_accuracy(df, ground_truth_column, predicted_column)
    # error_rate = get_transition_error_rate(df, ground_truth_column, predicted_column)

    ground_truth_segments = get_segments(ground_truth_transitions, len(df)) 
    predicted_segments = get_segments(predicted_transitions, len(df))
    greedy_iou = get_greedy_iou(ground_truth_segments, predicted_segments)

    ground_truth_sequence = get_segmentation_sequence(ground_truth_transitions, len(df))
    predicted_sequence = get_segmentation_sequence(predicted_transitions, len(df))

    # Populate the sequence with 0s and 1s
    gt_binary = [int(x) for x in ground_truth_sequence]
    pred_binary = [int(x) for x in predicted_sequence]

    GT_BINARY_COLUMN = "ground_truth_binary"
    PRED_BINARY_COLUMN = "predicted_binary"
    df[GT_BINARY_COLUMN] = gt_binary
    df[PRED_BINARY_COLUMN] = pred_binary

    k = get_k(df, ground_truth_column, unit="line")
    time_k = get_k(df, ground_truth_column, unit="time")

    df[constants.START_TIME] = pd.to_datetime(df[constants.START_TIME])
    df[constants.END_TIME] = pd.to_datetime(df[constants.END_TIME])

    # WindowDiff
    wd = windowdiff(ground_truth_sequence, predicted_sequence, k)
    weighted_wd = windowdiff(ground_truth_sequence, predicted_sequence, k, weighted=True)
    time_wd = time_windowdiff(
        df, 
        ground_truth_binary_column=GT_BINARY_COLUMN, 
        predicted_binary_column=PRED_BINARY_COLUMN, 
        time_k=time_k, 
        line_k=k, 
        weighted=False)
    weighted_time_wd = time_windowdiff(
        df, 
        ground_truth_binary_column=GT_BINARY_COLUMN, 
        predicted_binary_column=PRED_BINARY_COLUMN, 
        time_k=time_k, 
        line_k=k,
        weighted=True)

    # Pk
    p_k = pk(ground_truth_sequence, predicted_sequence)
    time_p_k = time_pk(df, ground_truth_binary_column=GT_BINARY_COLUMN, predicted_binary_column=PRED_BINARY_COLUMN, time_k=time_k, line_k=k)

    return {
        "segmentation/windowdiff": wd,
        "segmentation/weighted_windowdiff": weighted_wd,
        "segmentation/pk": p_k,
        "segmentation/time_windowdiff": time_wd,
        "segmentation/weighted_time_windowdiff": weighted_time_wd,
        "segmentation/time_pk": time_p_k,
        # "segmentation/greedy_iou": greedy_iou,
        # "segmentation/recall": recall,
        # "segmentation/precision": precision,
        # "segmentation/accuracy": accuracy,
        # "segmentation/error_rate": error_rate,
        # **transition_variables
    }
    
def calculate_retrieval_scores(df, ground_truth_transition_column, ground_truth_question_key, pred_question_key):
    """
    Get the transition indices. Look at the question_id for each transition.

    Compute the accuracy, precision, recall, and f1 score for the retrieval task
    """
    ground_truth_transition_indices = get_transitions(df, ground_truth_transition_column, as_indices=True, include_first=True) # We want the first segment
    
    # Get the question_id for each transition
    ground_truth_questions = np.array([df.iloc[i][ground_truth_question_key] for i in ground_truth_transition_indices])
    predicted_questions = np.array([df.iloc[i][pred_question_key] for i in ground_truth_transition_indices])

    is_correct = ground_truth_questions == predicted_questions
    accuracy = is_correct.mean()
    return {"retrieval/accuracy": accuracy}