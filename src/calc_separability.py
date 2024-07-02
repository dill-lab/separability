import json
import argparse
import pandas as pd
import numpy as np

from alignment_utils import get_score

from tqdm import tqdm

parser = argparse.ArgumentParser()


def normalize(x: float, min_val: float, max_val: float):
    return (x - min_val) / (max_val - min_val)


if __name__ == "__main__":
    parser.add_argument("--test_file", required=True, type=str, help="Path to the test file (in CSV format)")
    parser.add_argument("--modelA_name", required=True, type=str, help="Name of model A")
    parser.add_argument("--modelB_name", required=True, type=str, help="Name of model B")
    parser.add_argument("--modelA_gen_file", required=True, type=str, help="Path to the generation file of model A")
    parser.add_argument("--modelB_gen_file", required=True, type=str, help="Path to the generation file of model B")
    parser.add_argument("--gen_type", type=str, default="summaries",
                        help="Type of generation to consider in the generation file")
    parser.add_argument("--src_column", type=str, default="article", help="Name of the source column in the test file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to consider for alignment calculations")
    parser.add_argument("--metrics", type=str,
                        default="bertscore,cosine_sim,bertscore_length,entity_similarity,rouge_1",
                        help="Comma-separated list of metrics to use for alignment calculations")
    parser.add_argument("--out_file", type=str, help="Path to the output file")

    args = parser.parse_args()
    metrics = args.metrics.split(",")

    with open(args.modelA_gen_file, 'r') as f:
        generations1 = json.load(f)

    with open(args.modelB_gen_file, 'r') as f:
        generations2 = json.load(f)

    test_df = pd.read_csv(args.test_file, usecols=[args.src_column])

    sources = test_df[args.src_column].tolist()

    for metric in metrics:
        test_df[f"mean_{metric}_cross_alignment"] = 0
        test_df[f"mean_{metric}_self_alignment_{args.modelA_name}"] = 0
        test_df[f"mean_{metric}_self_alignment_{args.modelB_name}"] = 0

    # self-alignment args.modelA_name
    for metric in metrics:
        print(f"Calculating {metric} self-alignment for {args.modelA_name}")
        for idx, (summ_set_1, summ_set_2) in tqdm(enumerate(zip(generations1, generations1))):
            score_matrix = np.zeros((args.k_samples, args.k_samples))
            for j1, cand1 in enumerate(summ_set_1[args.gen_type][:args.k_samples]):
                for j2, cand2 in enumerate(summ_set_2[args.gen_type][:args.k_samples]):
                    if j1 < j2:
                        score = get_score(metric, cand1, cand2)
                        score_matrix[j1][j2] = score_matrix[j2][j1] = score

            np.fill_diagonal(score_matrix, np.nan)
            test_df[f"mean_{metric}_self_alignment_{args.modelA_name}"].iloc[idx] = np.nanmean(score_matrix)

    # self-alignment args.modelB_name
    for metric in metrics:
        print(f"Calculating {metric} self-alignment for {args.modelB_name}")
        for idx, (summ_set_1, summ_set_2) in tqdm(enumerate(zip(generations2, generations2))):
            score_matrix = np.zeros((args.k_samples, args.k_samples))
            for j1, cand1 in enumerate(summ_set_1[args.gen_type][:args.k_samples]):
                for j2, cand2 in enumerate(summ_set_2[args.gen_type][:args.k_samples]):
                    if j1 < j2:
                        score = get_score(metric, cand1, cand2)
                        score_matrix[j1][j2] = score_matrix[j2][j1] = score

            np.fill_diagonal(score_matrix, np.nan)
            test_df[f"mean_{metric}_self_alignment_{args.modelB_name}"].iloc[idx] = np.nanmean(score_matrix)

    for metric in metrics:
        print(f"Calculating {metric} cross-alignment for {args.modelA_name} and {args.modelB_name}")
        for idx, (summ_set_1, summ_set_2) in tqdm(enumerate(zip(generations1, generations2))):
            score_matrix = np.zeros((args.k_samples, args.k_samples))
            for j1, cand1 in enumerate(summ_set_1[args.gen_type][:args.k_samples]):
                for j2, cand2 in enumerate(summ_set_2[args.gen_type][:args.k_samples]):
                    if j1 <= j2:
                        score = get_score(metric, cand1, cand2)
                        score_matrix[j1][j2] = score_matrix[j2][j1] = score

            test_df[f"mean_{metric}_cross_alignment"].iloc[idx] = np.mean(score_matrix)

    for metric in metrics:
        test_df[f"{metric}_separability"] = test_df.apply(
            lambda x: max(x[f"mean_{metric}_self_alignment_{args.modelA_name}"],
                          x[f"mean_{metric}_self_alignment_{args.modelB_name}"])
                      - x[f"mean_{metric}_cross_alignment"], axis=1)

    for metric in metrics:
        min_alignment = np.min([test_df[f"mean_{metric}_self_alignment_{args.modelA_name}"].min(),
                                test_df[f"mean_{metric}_self_alignment_{args.modelB_name}"].min(),
                                test_df[f"mean_{metric}_cross_alignment"].min()])
        max_alignment = np.max([test_df[f"mean_{metric}_self_alignment_{args.modelA_name}"].max(),
                                test_df[f"mean_{metric}_self_alignment_{args.modelB_name}"].max(),
                                test_df[f"mean_{metric}_cross_alignment"].max()])

        test_df["normalized_self1"] = test_df.apply(
            lambda x: normalize(x[f"mean_{metric}_self_alignment_{args.modelA_name}"],
                                min_alignment,
                                max_alignment), axis=1)
        test_df["normalized_self2"] = test_df.apply(
            lambda x: normalize(x[f"mean_{metric}_self_alignment_{args.modelB_name}"],
                                min_alignment,
                                max_alignment), axis=1)
        test_df["normalized_cross"] = test_df.apply(lambda x: normalize(x[f"mean_{metric}_cross_alignment"],
                                                                        min_alignment,
                                                                        max_alignment), axis=1)

        test_df[f"{metric}_separability"] = test_df.apply(lambda x: max(x["normalized_self1"],
                                                                        x[f"normalized_self2"])
                                                                    - x["normalized_cross"], axis=1)

        test_df.drop(columns=["normalized_self1", "normalized_self2", "normalized_cross"], inplace=True)

    test_df.to_csv(args.output_file)
