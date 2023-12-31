import pandas as pd

N_QUESTIONS = 201
RECALL_COL = "Must_have"


def compute_recall(df: pd.DataFrame, column: str, metric: str) -> pd.DataFrame:
    """Compute an approximate version of recall"""
    misses = []
    col_name = f"avg_{column}_{metric}"
    recall = df[col_name]
    for index, row in df.iterrows():
        miss = [i[1] for i in row[f"raw_label_{column}_{metric}"] if "True" not in i[0]]
        misses.append(miss)
    return pd.DataFrame({f"recall_{column}": recall, "misses": misses})


def compute_precision(df: pd.DataFrame, columns: list, metric: str) -> pd.DataFrame:
    """Compute an approximate version of precision"""
    contras = []
    col_names = [f"sum_{column}_{metric}" for column in columns]
    n_cols = [f"n_{column}_{metric}" for column in columns]
    precision = 1 - (df.loc[:, col_names].sum(axis=1) / df.loc[:, n_cols].sum(axis=1))
    for index, row in df.iterrows():
        contra = [
            i[1]
            for column in columns
            for i in row[f"raw_label_{column}_{metric}"]
            if "True" in i[0]
        ]
        contras.append(contra)
    return pd.DataFrame({f"precision_{columns}": precision, "contra": contras})


def compute_overall_metrics(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Compute Precision and Recall on the answer level"""
    precision = compute_precision(df, columns=cols, metric="is_contradict")
    df = pd.concat([df, precision], axis=1)
    df.rename(
        columns={
            f"avg_{RECALL_COL}_is_entails": "recall",
            f"precision_{cols}": "precision",
        },
        inplace=True,
    )
    return df


def compute_macro_scores(df: pd.DataFrame) -> dict:
    """Compute macro-level metrics"""
    contra = df.contra.apply(lambda x: len(x)).sum()
    results = {
        "Comp": df.recall.sum() / N_QUESTIONS,
        "Comp_norm": df.recall.mean(),
        "Hall": contra / N_QUESTIONS,
        "Hall_norm": contra / len(df),
        "N_contra": int(contra),
        "N_answered_questions": len(df),
    }
    return results
