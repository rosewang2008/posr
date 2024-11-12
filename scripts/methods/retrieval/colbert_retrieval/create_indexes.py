import os
import pandas as pd
import sys
sys.path.append(os.getcwd())
from scripts import constants
from scripts.methods.utils import get_problem_text, get_question_id
from ragatouille import RAGPretrainedModel


MATH_DIR = os.path.join(constants.OCR_DIR, constants.MATH_DIR)

def create_index(problem_set_texts, problem_set, document_ids):
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    index_path = RAG.index(index_name=problem_set,
                           collection=problem_set_texts,
                           document_ids=document_ids)
    print(index_path)


if __name__ == "__main__":
    for problem_set in os.listdir(MATH_DIR):
        if len(problem_set) == 3:
            problem_set_dir = os.path.join(MATH_DIR, problem_set)
            problems = [problem for problem in os.listdir(problem_set_dir)
                        if os.path.splitext(problem)[1].lower() == ".txt"]
            problem_set_texts = [get_problem_text(os.path.join(problem_set_dir, problem))
                                 for problem in problems]
            document_ids = [get_question_id(problem) for problem in problems]
            create_index(problem_set_texts, problem_set, document_ids)
    # IMPORTANT: after running, move ".ragatouille" folder to "scripts/methods/retrieval/colbert_retrieval"