import itertools
import pandas as pd


def create_df_reasoning(df: pd.DataFrame, tasks: list, cols: list) -> pd.DataFrame:
    """Collect the reasoning behind every answer"""
    list_of_dfs = []
    for ind, col in enumerate(cols):
        res_dict = {}
        for att in tasks:
            col_name = f"label_{col}_{att}"
            raw_col_name = f"raw_label_{col}_{att}"

            n_att_per_q = [len(i) for i in df[col_name]]
            res_dict["question"] = list(
                itertools.chain(
                    *[
                        i * [df["Question"].iloc[ind]]
                        for ind, i in enumerate(n_att_per_q)
                    ]
                )
            )
            res_dict["answer"] = list(
                itertools.chain(
                    *[i * [df["result"].iloc[ind]] for ind, i in enumerate(n_att_per_q)]
                )
            )
            res_dict["statement"] = [
                row[1] for i in range(len(df)) for row in df[raw_col_name][i]
            ]

            res_dict[f"label_{att}"] = list(itertools.chain(*[i for i in df[col_name]]))

            res_dict[f"reasoning_{att}"] = [
                row[0] for i in range(len(df)) for row in df[raw_col_name][i]
            ]

        res_dict["category"] = [col] * sum(n_att_per_q)
        list_of_dfs.append(pd.DataFrame(res_dict))
    df_reasoning = pd.concat(list_of_dfs, axis=0).sort_values(
        by=["question", "category"]
    )
    return df_reasoning
