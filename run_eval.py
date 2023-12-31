import os
import asyncio
from asyncio import to_thread
import numpy as np
from evaluation.metrics import compute_overall_metrics, compute_macro_scores
from evaluation.utils import create_df_reasoning

import argparse
import pandas as pd
import json

from openai.error import InvalidRequestError
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate

COLS = ["Must_have", "Nice_to_have"]
COLS_TO_SAVE = ["Question", "result", "recall", "precision", "contra"]
TASKS = ["is_entails", "is_contradict"]


async def run_experiment(
    results_file: str,
    version_name: str,
):
    global ans_results
    results_dict, ans_results_dict = {}, {}
    list_of_metric_dfs = []

    # Load ground-truth data including statements
    df_gt = pd.read_json(
        "dataset/questions_w_answers.jsonl", orient="records", lines=True
    )

    results = pd.read_json(results_file, orient="records")
    results = df_gt.loc[:, ["Question", "Must_have", "Nice_to_have"]].merge(
        results, on="Question", how="inner"
    )
    print(f"Evaluated questions: {len(results)} (out of {len(df_gt)})\n-----")

    # Evaluate each statement
    for col_name in COLS:
        for task in TASKS:
            print(f"Computing for task: {task}, col_name: {col_name}")
            await asyncio.gather(
                *[
                    run_single_nli_answer(idx, row, ans_results_dict, col_name, task)
                    for idx, row in results.iterrows()
                ],
                return_exceptions=True,
            )
            metric_res = pd.DataFrame(ans_results_dict).T
            list_of_metric_dfs.append(metric_res)
        ans_results = pd.concat(list_of_metric_dfs, axis=1)
    results = pd.merge(results, ans_results, left_index=True, right_index=True)

    # Compute overall metrics
    results = compute_overall_metrics(results, cols=COLS)
    macro_results = compute_macro_scores(results)
    print(f"Macro results:\n{json.dumps(macro_results, indent=4)}")

    # Save results
    print("Saving results...")
    if not os.path.exists(version_name):
        os.makedirs(version_name)
    with open(f"{version_name}/macro_results.json", "w") as fp:
        json.dump(macro_results, fp)
    results.loc[:, COLS_TO_SAVE].to_csv(f"{version_name}/dataset_statistics.csv")
    results = pd.DataFrame(results).reset_index(drop=True)
    df_reasoning = create_df_reasoning(results, TASKS, COLS)
    df_reasoning.to_csv(f"{version_name}/reasoning.csv")


async def run_single_nli_answer(idx, row, ans_results_dict, col_name, task):
    answers, labels = row[col_name], []
    if (task == "is_entails") & (col_name == "Nice_to_have"):
        ans_results_dict[idx] = {
            f"raw_label_{col_name}_{task}": ["False"] * len(answers),
            f"label_{col_name}_{task}": [False] * len(answers),
            f"avg_{col_name}_{task}": 0,
            f"sum_{col_name}_{task}": 0,
            f"n_{col_name}_{task}": 0,
        }
    else:
        with open(f"evaluation/prompts/{task}.txt", "r") as f:
            prompt_template = f.read()
        prompt = PromptTemplate.from_template(prompt_template)
        ans_checker = LLMChain(llm=LLM, prompt=prompt, output_key="label")
        for answer in answers:
            try:
                res = await to_thread(
                    ans_checker,
                    {
                        "llm_answer": row["result"],
                        "answer": answer,
                        "question": row["Question"],
                    },
                )
                labels.append((res["label"], answer))

            except InvalidRequestError as e:
                print(
                    f"Invalid request error -- {e} \n"
                    f"Question: {row['Question']}, Hypothesis: {answer}"
                )
                labels.append((res["label"], answer))

        ans_true = ["true" in label[0].lower()[-20:] for label in labels]
        ans_results_dict[idx] = {
            f"raw_label_{col_name}_{task}": labels,
            f"label_{col_name}_{task}": ans_true,
            f"avg_{col_name}_{task}": np.mean(ans_true),
            f"sum_{col_name}_{task}": np.sum(ans_true),
            f"n_{col_name}_{task}": len(labels),
        }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="K-QA Benchmark Evaluation")
    parser.add_argument("--result_file", type=str, default="dummy_res.json")
    parser.add_argument("--version", type=str, default="dummy_exp")
    parser.add_argument(
        "--on_openai",
        type=bool,
        default=True,
        help="Calling GPT-4 model from Azure or from OpenAI",
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.on_openai:
        LLM = AzureChatOpenAI(
            deployment_name="gpt-4-8k",
            model_name="gpt-4-8k",
            temperature=0,
            max_retries=20,
            verbose=True,
        )
    else:
        LLM = ChatOpenAI(
            deployment_name="gpt-4",
            model_name="gpt-4",
            temperature=0,
            max_retries=20,
            verbose=True,
        )
    asyncio.run(
        run_experiment(
            results_file=args.result_file,
            version_name=args.version,
        )
    )
