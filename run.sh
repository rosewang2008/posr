#!/bin/bash

# SEGMENTATION 

# Top-k word segmentation
ks=("10" "20")
for k in "${ks[@]}"; do
    python3 scripts/run.py --task=segmentation --method=top_k_words --k=${k} --run_inference --run_evaluation
done

# TextTiling 
python3 scripts/run.py --task=segmentation --method=texttiling --run_inference --run_evaluation

# Topic segmentation
python3 scripts/run.py --task=segmentation --method=topic_segmentation --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation --method=topic_segmentation --encoder_model=all-mpnet-base-v2 --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation --method=topic_segmentation --encoder_model=all-MiniLM-L12-v2 --run_inference --run_evaluation

# Stage segmentation
modes=("average" "max")
for mode in "${modes[@]}"; do
    python3 scripts/run.py --task=segmentation --method=stage_segmentation --run_inference --run_evaluation --stage_segmentation_mode=${mode}
    python3 scripts/run.py --task=segmentation --method=stage_segmentation --encoder_model=all-mpnet-base-v2 --run_inference --run_evaluation --stage_segmentation_mode=${mode}
    python3 scripts/run.py --task=segmentation --method=stage_segmentation --encoder_model=all-MiniLM-L12-v2 --run_inference --run_evaluation --stage_segmentation_mode=${mode}
done

# Zero-shot prompting LLMs
python3 scripts/run.py --task=segmentation --method=gpt-4 --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation --method=claude-haiku --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation --method=claude-sonnet --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation --method=claude-opus --run_inference --run_evaluation

# RETRIEVAL

# Jaccard similarity
python3 scripts/run.py --task=retrieval --method=jaccard_similarity --run_inference --run_evaluation

# TF-IDF
python3 scripts/run.py --task=retrieval --method=tfidf --run_inference --run_evaluation

# BM-25
python3 scripts/run.py --task=retrieval --method=bm25 --run_inference --run_evaluation

# Colbert
python3 scripts/run.py --task=retrieval --method=colbert_retrieval --run_inference --run_evaluation

# Zero-shot prompting LLMs
python3 scripts/run.py --task=retrieval --method=gpt-4 --run_inference --run_evaluation --retrieve_segment_by_segment
python3 scripts/run.py --task=retrieval --method=claude-haiku --run_inference --run_evaluation --retrieve_segment_by_segment
python3 scripts/run.py --task=retrieval --method=claude-sonnet --run_inference --run_evaluation --retrieve_segment_by_segment
python3 scripts/run.py --task=retrieval --method=claude-opus --run_inference --run_evaluation --retrieve_segment_by_segment

# SEGMENTATION AND RETRIEVAL

# Best independent segmentation method -> each retrieval method
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->jaccard_similarity" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->tfidf" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->bm25" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->colbert_retrieval" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->gpt-4" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->claude-haiku" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->claude-sonnet" --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method="claude-opus->claude-opus" --run_inference --run_evaluation

# Zero-shot prompting LLMs, i.e. POSR methods that perform segmentation and retrieval jointly
python3 scripts/run.py --task=segmentation_and_retrieval --method=gpt-4 --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method=claude-haiku --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method=claude-sonnet --run_inference --run_evaluation
python3 scripts/run.py --task=segmentation_and_retrieval --method=claude-opus --run_inference --run_evaluation