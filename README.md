# Separability

This repository contains the code for the paper **Compare without Despair: Reliable Preference Evaluation with Generation Separability** 
by Sayan Ghosh, Tejas Srinivasan, and Swabha Swayamdipta

**Repository under construction to add more features and optimizations, please contact ghoshsay@usc.edu for any queries in the meantime**

## Abstract
Human evaluation of generated language through pairwise preference judgments is pervasive. However, under common scenarios, such as when generations from a model pair are very similar, or when stochastic decoding results in large variations in generations, it results in inconsistent preference ratings. We address these challenges by introducing a meta-evaluation measure, separability, which estimates how suitable a test instance is for pairwise preference evaluation. For a candidate test instance, separability samples multiple generations from a pair of models, and measures how distinguishable the two sets of generations are. Our experiments show that instances with high separability values yield more consistent preference ratings from both human- and auto-raters. Further, the distribution of separability allows insights into which test benchmarks are more valuable for comparing models. Finally, we incorporate separability into ELO ratings, accounting for how suitable each test instance might be for reliably ranking LLMs. Overall, separability has implications for consistent, efficient and robust preference evaluation of LLMs with both human- and auto-raters.

## How to Compute Separability
Suppose you have two models, A and B, and you want to compute the separability of a test set with respect to this model pair.
1. Install the requirements using `pip install -r requirements.txt`
2. For a specific test set, ensure model generations are contained in JSONL files with the following format:
```json
[
    {
      "summaries": [
          "summary1",
          "summary2",
        ...
      ]    
    },
    { 
      "summaries": [
          "summary1",
          "summary2",
        ...
      ]    
    },
  ...
]
```
"Summaries" can be replaced with any generation type.
(Note: The order of test instances should be the same order for both models)
The test set itself must be in csv format (only one column including the source text is necessary)

3. Run a command of the following example to compute the separability of the test set:
```bash
python compute_separability.py \
--input_file <path_to_input_file> \
--modelA_gen_file <path_to_modelA_gen_file> \
--modelB_gen_file <path_to_modelB_gen_file> \
--modelA_name <modelA_name> \
--modelB_name <modelB_name> \
--gen_type "summaries" \
--src_column "article" \
--num_samples 5 \
--metrics "bertscore_length,rouge_1" \
--output_file <path_to_output_file> 
```

The current supported metrics are: BERTScore and length-adjusted BERTScore ("bertscore,bertscore_length")
ROUGE-1-F1 ("rouge_1"), entity similarity ("entity_sim"), BLEU ("bleu"), and cosine similarity ("cosine_sim")

The output CSV will include self-alignments, cross alignments, and separability scores for each test instance.

## Human Preference Data
We provide (anonymized) data corresponding to our human evaluation experiments in the `data/preference_data/` directory. 

## Citation 
