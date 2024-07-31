import evaluate
import spacy
import re
import numpy as np

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

model = SentenceTransformer("all-distilroberta-v1")
model.max_seq_length = 200

nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()


def entity_sim(gen_a: str, gen_b: str) -> float:
    gen_a_ents = set()
    gen_b_ents = set()

    doc_a = nlp(gen_a)
    for ent in doc_a.ents:
        ent_text = re.sub('[^0-9 a-zA-Z]+', "", ent.text)
        ent_text = lemmatizer.lemmatize(ent_text.replace("the", "").strip())
        ent_text = ent_text.lower()

        gen_a_ents.add(ent_text)

    doc_b = nlp(gen_b)
    for ent in doc_b.ents:
        ent_text = re.sub('[^0-9 a-zA-Z]+', "", ent.text)
        ent_text = lemmatizer.lemmatize(ent_text.replace("the", "").strip())
        ent_text = ent_text.lower()

        gen_b_ents.add(ent_text)

    intersection = len(gen_a_ents.intersection(gen_b_ents))
    union = (len(gen_b_ents) + len(gen_b_ents)) - intersection

    # If there are no entities in either of the generations, return 1
    if union == 0:
        return 1

    return intersection / union


def calc_bertscore(gen_a: str, gen_b: str) -> float:
    return bertscore.compute(predictions=[gen_a], references=[gen_b], lang='en')['f1'][0]


def calc_bertscore_length(gen_a: str, gen_b: str) -> float:
    ref, cand = (gen_a, gen_b) if len(gen_a) > len(gen_b) else (gen_b, gen_a)
    try:
        length_pen = np.exp(1 - len(word_tokenize(ref)) / len(word_tokenize(cand)))
        return length_pen * calc_bertscore(gen_a, gen_b)
    except ZeroDivisionError:
        print(f"{gen_a} \n {gen_b}")


def calc_rouge(gen_a: str, gen_b: str) -> float:
    return rouge.score(prediction=gen_a, target=gen_b)["rouge1"].fmeasure


def calc_bleu(gen_a: str, gen_b: str) -> float:
    # If using a length penalty, the shorter generation is the candidate
    ref, cand = (gen_a, gen_b) if len(gen_a) > len(gen_b) else (gen_b, gen_a)
    return bleu.compute(predictions=[cand], references=[ref])["bleu"]


def cosine_sim(gen_a: str, gen_b: str) -> float:
    embeddings1 = model.encode([gen_a], convert_to_tensor=True)
    embeddings2 = model.encode([gen_b], convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores[0]


def get_score(metric: str, gen_a: str, gen_b: str) -> float:
    metric_funcs = {
        "rouge_1": calc_rouge,
        "bertscore": calc_bertscore,
        "bertscore_length": calc_bertscore_length,
        "entity_sim": entity_sim,
        "bleu": calc_bleu,
        "cosine_sim": cosine_sim
    }

    if metric in metric_funcs:
        return metric_funcs[metric](gen_a, gen_b)
    else:
        raise ValueError(f"Metric {metric} not supported")
