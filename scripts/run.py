"""
Main inference and evaluations script
"""

import pandas as pd
import os
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())

from scripts import (
    constants,
    metrics
)

from methods.segmentation import (
    simple_regex,
    texttiling,
    topic_segmentation,
    stage_segmentation,
    gpt4 as seg_gpt4,
    claude as seg_claude,
)

from methods.segmentation.C99 import C99

from methods.retrieval import (
    ngrams,
    gpt4 as ret_gpt4,
    claude as ret_claude
)
from methods.retrieval.colbert_retrieval import colbert_retrieval

from methods.segmentation_and_retrieval import (
    gpt4 as segret_gpt4,
    claude as segret_claude
)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_inference", action="store_true", help="Run inference calls")
parser.add_argument("--run_evaluation", action="store_true", help="Run evaluation calls")
parser.add_argument("--task", type=str, help="segmentation, retrieval, or segmentation_and_retrieval")
parser.add_argument("--method", type=str, help="method as NL; if task is segmentation_and_retrieval, then method shd be formatted as [segmentation_method]->[retrieval_method]")
parser.add_argument("--k", type=int, default=10, help="int k for segmentation by top k words")
parser.add_argument("--encoder_model", type=str, help="for topic segmentation from: https://sbert.net/docs/pretrained_models.html", default="bert-base-nli-stsb-mean-tokens")
parser.add_argument("--stage_segmentation_mode", type=str, help="for HMM number of segments -- either the average or the max number of segments", default="average")
parser.add_argument("--retrieve_segment_by_segment", action="store_true", help="Make LLMs retrieve for one segment at a time, rather than all segments at once.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

args = parser.parse_args()

def get_task_dir(task): 
    if task == constants.SEGMENTATION_TASK:
        return constants.SEGMENTATION_DIR
    
    elif task == constants.RETRIEVAL_TASK:
        return constants.RETRIEVAL_DIR
    
    elif task == constants.SEGMENTATION_AND_RETRIEVAL_TASK:
        return constants.SEGMENTATION_AND_RETRIEVAL_DIR

def get_segmentation_method(method):
    if '->' in method:
        return method.split('->')[0]
    else:
        return method
    
def get_retrieval_method(method):
    if '->' in method:
        return method.split('->')[1]
    else:
        return method
    
def get_prompt_num(args, retrieval=False):
    if retrieval and args.retrieve_segment_by_segment:
        return 2
    else:
        return 1
    
def get_segmentation_method_dir(args):
    if args.task == constants.SEGMENTATION_TASK:
        method = args.method
    elif args.task == constants.SEGMENTATION_AND_RETRIEVAL_TASK and '->' in args.method:
        method = get_segmentation_method(args.method)

    if method == "top_k_words":
        method_dir = f'{args.method}_{str(args.k)}'
    elif method == 'topic_segmentation':
        method_dir = f"{args.method}_{args.encoder_model}"
    elif method == "stage_segmentation":
        method_dir = f"{args.method}_{args.encoder_model}_{args.stage_segmentation_mode}"
    else:
        method_dir = method

    return method_dir

def get_segmentation_and_retrieval_method_dir(method):
    assert '->' in method, "the method should be formatted as [segmentation_method]->[retrieval_method]"
    retrieval_method = method.split('->')[0]
    segmentation_method = method.split('->')[1]

    if retrieval_method == 'top_k_words':
        retrieval_method = f'{retrieval_method}_{str(args.k)}'
    if segmentation_method == 'top_k_words':
        segmentation_method = f'{segmentation_method}_{str(args.k)}'

    return f'{retrieval_method}->{segmentation_method}'

def get_method_dir(args):
    if args.task == constants.SEGMENTATION_TASK:
        method_dir = get_segmentation_method_dir(args)
    elif args.task == constants.RETRIEVAL_TASK:
        method_dir = args.method
    elif args.task == constants.SEGMENTATION_AND_RETRIEVAL_TASK:
        if '->' in args.method: # independent seg -> independent ret
            method_dir = get_segmentation_and_retrieval_method_dir(args.method)
        else: # joint posr
            method_dir = args.method
    return method_dir

######################

def get_test_fname_list():
    test_transcript_fnames = [f for f in os.listdir(constants.TEST_DIR) if f.endswith(".csv")]
    return test_transcript_fnames

def get_evaluation_df(args):
    run_segmentation = (constants.SEGMENTATION_TASK in args.task)
    run_retrieval = (constants.RETRIEVAL_TASK in args.task)
    transcript_dir = get_output_dir(args)

    eval_path = os.path.join(os.path.dirname(transcript_dir), "results.csv")
    if os.path.exists(eval_path) and not args.overwrite:
        results_df = pd.read_csv(eval_path)
        return results_df

    test_transcript_fnames = get_test_fname_list()
    results = []
    for transcript_fname in tqdm(os.listdir(transcript_dir)):
        result = {}
        transcript_fpath = os.path.join(transcript_dir, transcript_fname)
        if not transcript_fname.endswith(".csv"):
            continue

        # Make sure we only evaluate on test transcripts
        if transcript_fname not in test_transcript_fnames:
            continue
        transcript_df = pd.read_csv(transcript_fpath)

        # bucket -1 and null question ids together, by turning them all into -1
        transcript_df[constants.ACTUAL_QUESTION_KEY] = transcript_df[constants.ACTUAL_QUESTION_KEY].fillna(-1)
        transcript_df[constants.PRED_QUESTION_KEY] = transcript_df[constants.PRED_QUESTION_KEY].fillna(-1)

        result[constants.TRANSCRIPT_KEY] = transcript_fname

        if run_segmentation:
            segmentation_metrics = metrics.calculate_segmentation_scores(
                df=transcript_df,
                ground_truth_column=constants.ACTUAL_SEGMENT_KEY,
                predicted_column=constants.PRED_SEGMENT_KEY
            )
            result.update(segmentation_metrics)

        if run_retrieval:
            retrieval_metrics = metrics.calculate_retrieval_scores(
                df=transcript_df,
                ground_truth_transition_column=constants.ACTUAL_SEGMENT_KEY,
                ground_truth_question_key=constants.ACTUAL_QUESTION_KEY,
                pred_question_key=constants.PRED_QUESTION_KEY
                )
            both_metrics = metrics.calculate_segmentation_retrieval_scores(
                transcript_df, 
                ground_truth_question_column=constants.ACTUAL_QUESTION_KEY,
                predicted_question_column=constants.PRED_QUESTION_KEY
                )
            result.update(retrieval_metrics)
            result.update(both_metrics)

        # if method is gpt-4 or claude, add model usage metrics
        if 'gpt-4' in args.method or 'claude' in args.method:
            model_usage_metrics = metrics.calculate_model_usage_metrics(transcript_df, args.method, args.task, transcript_fname) 
            result.update(model_usage_metrics)

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_path, index=False)
    return results_df

def print_evaluation_results(results_df, args):
    transcript_dir = get_output_dir(args)
    eval_path = os.path.join(os.path.dirname(transcript_dir), "results.txt")

    # Check that results_df only has test_fnames
    test_transcript_fnames = get_test_fname_list()
    # Only keep test transcripts
    results_df = results_df[results_df[constants.TRANSCRIPT_KEY].isin(test_transcript_fnames)]
    try:
        assert all([transcript_fname in test_transcript_fnames for transcript_fname in results_df[constants.TRANSCRIPT_KEY]])
    except AssertionError:
        import pdb; pdb.set_trace()

    try:
        assert len(test_transcript_fnames) == len(results_df)
        assert len(results_df) == 270
    except AssertionError:
        import pdb; pdb.set_trace()

    with open(eval_path, "w") as f:
        f.write(f"Task: {args.task}, from {transcript_dir}, with n={len(results_df)}\n")
        print(f"Task: {args.task}, from {transcript_dir}, with n={len(results_df)}")
        for key in results_df.keys():
            if key == constants.TRANSCRIPT_KEY: continue
            mean = round(results_df[key].mean(), 3)
            std = round(results_df[key].std(), 3)
            f.write(f"{key}: ${mean} \pm {std}$\n")
            print(f"{key}: ${mean} \pm {std}$")

def get_output_dir(args):
    task_dir = get_task_dir(args.task)
    method_dir = get_method_dir(args)
    output_dir = os.path.join(constants.RESULTS_DIR, task_dir, method_dir, "transcripts")

    if args.method == 'topic_segmentation':
        output_dir = os.path.join(
            constants.RESULTS_DIR,
            constants.SEGMENTATION_DIR,
            f"{args.method}_{args.encoder_model}",
            "transcripts")

    if args.method == "stage_segmentation":
        output_dir = os.path.join(
            constants.RESULTS_DIR,
            constants.SEGMENTATION_DIR,
            f"{args.method}_{args.encoder_model}_{args.stage_segmentation_mode}",
            "transcripts")

    return output_dir

def run_segmentation(args):
    print("Running segmentation...")
    method = get_segmentation_method(args.method)
    prompt_num = get_prompt_num(args)
    method_dir = get_segmentation_method_dir(args)
    transcript_dir = constants.TRANSCRIPT_DATA_DIR

    model = None
    for transcript_fname in tqdm(os.listdir(transcript_dir)):
        transcript_fpath = os.path.join(transcript_dir, transcript_fname)
        transcript_df = pd.read_csv(transcript_fpath)

        output_dir = os.path.join(constants.RESULTS_DIR, constants.SEGMENTATION_DIR, method_dir, "transcripts")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_fpath = os.path.join(output_dir, transcript_fname)
        if os.path.exists(output_fpath) and not args.overwrite:
            continue

        if method == 'manual_transition_words':
            transcript_df = simple_regex.segment_by_transition_words(transcript_df)
        elif method == 'top_k_words':
            transcript_df = simple_regex.segment_by_top_k(transcript_df, args.k)
        elif method == "texttiling":
            transcript_df = texttiling.segment(
                transcript_df, 
                text_column=constants.TEXT_KEY,
                output_column=constants.PRED_SEGMENT_KEY) 
        elif method == 'topic_segmentation':
            if model is None:
                model = C99(window = constants.C99_WINDOW, std_coeff = constants.C99_STD_COEFF)
            c99_model = model
            transcript_df = topic_segmentation.segment(transcript_df, c99_model=c99_model, encoder_model_name=args.encoder_model)
        elif method == 'stage_segmentation':
            # First fit HMM to train dataset
            num_components = constants.AVG_NUM_SEGMENTS_PER_TRANSCRIPT if args.stage_segmentation_mode == 'average' else constants.MAX_NUM_SEGMENTS_PER_TRANSCRIPT
            if model is None:
                model = stage_segmentation.load_and_fit_hmm_model(
                    encoder_name=args.encoder_model,
                    num_components=num_components
                )
            hmm_model = model
            transcript_df = stage_segmentation.segment(transcript_df, hmm_model=hmm_model, encoder_name=args.encoder_model)
        elif 'gpt-4' in method:
            transcript_df = seg_gpt4.segment(transcript_df, prompt_num)
        elif 'claude' in method:
            transcript_df = seg_claude.segment(transcript_df, prompt_num, method)
        else:
            raise ValueError(f"Invalid segmentation method: {method}")
        transcript_df.to_csv(output_fpath, index=False)

def run_retrieval(args):
    print("Running retrieval...")
    retrieval_method = get_retrieval_method(args.method)
    prompt_num = get_prompt_num(args, retrieval=True)

    # define (1) directory for loading segmented transcripts, (2) directory for outputting annotated transcripts, and (3) segment key, depending on task
    if args.task == constants.RETRIEVAL_TASK:
        transcript_dir = constants.TRANSCRIPT_DATA_DIR 
        output_dir = os.path.join(constants.RESULTS_DIR, constants.RETRIEVAL_DIR, retrieval_method, "transcripts")
        segment_key = constants.ACTUAL_SEGMENT_KEY
    elif args.task == constants.SEGMENTATION_AND_RETRIEVAL_TASK:
        segmentation_method = get_segmentation_method(args.method)
        transcript_dir = os.path.join(constants.RESULTS_DIR, constants.SEGMENTATION_DIR, segmentation_method, "transcripts")
        output_dir = os.path.join(constants.RESULTS_DIR, constants.SEGMENTATION_AND_RETRIEVAL_DIR, f'{segmentation_method}->{retrieval_method}', "transcripts")
        segment_key = constants.PRED_SEGMENT_KEY

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for transcript_fname in tqdm(os.listdir(transcript_dir)):
        transcript_fpath = os.path.join(transcript_dir, transcript_fname)
        transcript_df = pd.read_csv(transcript_fpath)

        output_fpath = os.path.join(output_dir, transcript_fname)
        if os.path.exists(output_fpath) and not args.overwrite:
            continue

        if retrieval_method == 'jaccard_similarity' or retrieval_method == 'bm25' or retrieval_method == 'tfidf':
            transcript_df = ngrams.retrieve(transcript_df, retrieval_method, segment_key)
        elif retrieval_method == 'colbert_retrieval':
            transcript_df = colbert_retrieval.retrieve(transcript_df, segment_key)
        elif 'gpt-4' in retrieval_method:
            if args.retrieve_segment_by_segment:
                transcript_df = ret_gpt4.retrieve_segment_by_segment(transcript_df, segment_key, prompt_num)
            else:
                transcript_df = ret_gpt4.retrieve(transcript_df, segment_key, prompt_num)
        elif 'claude' in retrieval_method:
            if args.retrieve_segment_by_segment:
                transcript_df = ret_claude.retrieve_segment_by_segment(transcript_df, segment_key, prompt_num, retrieval_method)
            else:
                transcript_df = ret_claude.retrieve(transcript_df, segment_key, prompt_num, retrieval_method)
        transcript_df.to_csv(output_fpath, index=False)

def run_segmentation_and_retrieval(args):
    if '->' in args.method: # independent seg -> independent ret
        run_segmentation(args)
        run_retrieval(args) 
    else: # joint posr
        assert 'gpt-4' in args.method or 'claude' in args.method, "Invalid joint POSR method"
        prompt_num = get_prompt_num(args)
        transcript_dir = constants.TRANSCRIPT_DATA_DIR
        for transcript_fname in tqdm(os.listdir(transcript_dir)):
            transcript_fpath = os.path.join(transcript_dir, transcript_fname)
            transcript_df = pd.read_csv(transcript_fpath)

            output_dir = os.path.join(constants.RESULTS_DIR, constants.SEGMENTATION_AND_RETRIEVAL_DIR, args.method, "transcripts")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_fpath = os.path.join(output_dir, transcript_fname)
            if os.path.exists(output_fpath) and not args.overwrite:
                continue

            if 'gpt-4' in args.method:
                segret_gpt4.segment_and_retrieve(transcript_df, prompt_num)
            elif 'claude' in args.method:
                segret_claude.segment_and_retrieve(transcript_df, prompt_num, args.method)

            transcript_df.to_csv(output_fpath, index=False)

if __name__ == "__main__":
    valid_methods = [
        'gpt-4', 
        'claude-haiku', 'claude-sonnet', 'claude-opus', 
        'topic_segmentation', 'stage_segmentation',
        'manual_transition_words', 'texttiling', 'top_k_words', 
        'jaccard_similarity', 'bm25', 'tfidf',
        'colbert_retrieval'
    ]

    if '->' in args.method:
        segmentation_method = get_segmentation_method(args.method)
        retrieval_method = get_retrieval_method(args.method)
        assert segmentation_method in valid_methods, "Invalid segmentation method"
        assert retrieval_method in valid_methods, "Invalid retrieval method"
    else:
        assert args.method in valid_methods, "Invalid method"

    if args.retrieve_segment_by_segment:
        assert args.task == "retrieval" or (args.task == "segmentation_and_retrieval" and "->" in args.method), "--retrieve_segment_by_segment can only be used when task=retrieval, or when task=segmentation_and_retrieval involving an independent segmentation and an independent retrieval method, i.e. method=[segmentation_method]->[retrieval_method]"

    if args.run_inference: 
        if args.task == constants.SEGMENTATION_TASK:
            run_segmentation(args)
        elif args.task == constants.RETRIEVAL_TASK:
            run_retrieval(args)
        elif args.task == constants.SEGMENTATION_AND_RETRIEVAL_TASK:
            run_segmentation_and_retrieval(args)

    if args.run_evaluation:
        results_df = get_evaluation_df(args)
        print_evaluation_results(results_df, args)