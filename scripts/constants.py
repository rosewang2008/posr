SEGMENTATION_KEY = "time_segment"
RETRIEVAL_KEY = "question_id"
SPEAKER_KEY = "speaker_#"
TEXT_KEY = "text"
START_TIME = "startTimestamp"
END_TIME = "endTimestamp"
PROBLEM_SET_KEY = "problem_set"

SEGMENTATION_TASK = "segmentation"
RETRIEVAL_TASK = "retrieval"
SEGMENTATION_AND_RETRIEVAL_TASK = "segmentation_and_retrieval"
TRANSCRIPT_KEY = "transcript"
SYSTEM_PROMPT_KEY = "system_prompt"
USER_PROMPT_KEY = "user_prompt"
RAW_OUTPUT_KEY = "raw_output"
NUM_PROMPT_TOKENS_KEY = "num_prompt_tokens"
NUM_COMPLETION_TOKENS_KEY = "num_completion_tokens"
NUM_TOTAL_TOKENS_KEY = "num_total_tokens"

MODEL_2_INPUT_TOKEN_COST_1M = {
    "gpt-4": 10.00,
    "claude-opus": 15.00,
    "claude-sonnet": 3.00,
    "claude-haiku": 0.25,
}

MODEL_2_OUTPUT_TOKEN_COST_1M = {
    "gpt-4": 30.00,
    "claude-opus": 75.00,
    "claude-sonnet": 15.00,
    "claude-haiku": 1.25,
}

# DIRECTORIES
RESULTS_DIR = "results"
PROMPTS_DIR = "prompts"
SYSTEM_DIR = "system"
USER_DIR = "user"
ANALYSIS_RESULTS_DIR = "results/analysis"
SEGMENTATION_PROMPT_FILENAME = "seg{prompt_num}.txt"
RETRIEVAL_PROMPT_FILENAME = "ret{prompt_num}.txt"
SEGMENTATION_AND_RETRIEVAL_PROMPT_FILENAME = "segret{prompt_num}.txt"
TRANSCRIPT_DATA_DIR = "data/transcripts"
OCR_DIR = "data/ocr"
DATASET_STATS_FILENAME = "results/dataset_statistics.txt"
SEGMENTATION_DIR = "segmentation"
RETRIEVAL_DIR = "retrieval"
SEGMENTATION_AND_RETRIEVAL_DIR = "segmentation_and_retrieval"
MATH_DIR = "Math"
READING_WRITING_DIR = "R_W"

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
HUMAN_AGREEMENT_DIR = "data/human_agreement"

PRED_SEGMENT_KEY = "pred_segment"
ACTUAL_SEGMENT_KEY = "actual_segment"
PRED_QUESTION_KEY = "pred_question_id"
ACTUAL_QUESTION_KEY = "actual_question_id"

CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-sonnet-20240229"
CLAUDE_OPUS = "claude-3-opus-20240229"

LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"

COLBERT_INDEXES_DIR = "scripts/methods/retrieval/colbert_retrieval/.ragatouille/colbert/indexes"

C99_WINDOW = 4
C99_STD_COEFF = 1

TRAIN_EMBEDDINGS_FPATH = "scripts/methods/segmentation/{encoder_name}_train_embeddings.pkl"
HMM_NUM_ITER = 100
AVG_NUM_SEGMENTS_PER_TRANSCRIPT = round(11.920000)
MAX_NUM_SEGMENTS_PER_TRANSCRIPT = 25

RETRIEVAL_THRESHOLD_DIR = "results/retrieval_threshold"
SIMILARITY_SCORE_KEY = "score"
MIN_NUM_OF_PROBLEMS_IN_PROBLEM_SET = 10
JACCARD_RETRIEVAL_THRESHOLD = 0.11
TFIDF_RETRIEVAL_THRESHOLD = 0.4
BM25_RETRIEVAL_THRESHOLD = 0.19
COLBERT_RETRIEVAL_THRESHOLD = 0.14