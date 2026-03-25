import pandas as pd

from src.data_utils import choose_support_query_indices, load_samples_from_indices
from src.baseline_utils import train_baseline_on_support, evaluate_model


def run_baseline_episode(
    task_index,
    k_shot,
    target_size=(192, 192),
    seed=42,
    prefer_positive=True,
    epochs=8,
    batch_size=2,
    learning_rate=1e-3
):
    support_idx, query_idx = choose_support_query_indices(
        task_index=task_index,
        k_shot=k_shot,
        seed=seed,
        prefer_positive=prefer_positive
    )

    X_support, Y_support = load_samples_from_indices(task_index, support_idx, target_size)
    X_query, Y_query = load_samples_from_indices(task_index, query_idx, target_size)

    model, _ = train_baseline_on_support(
        X_support,
        Y_support,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=0
    )

    metrics = evaluate_model(model, X_query, Y_query)
    metrics["k_shot"] = k_shot
    metrics["task_name"] = task_index["task_name"]
    metrics["support_size"] = len(support_idx)
    metrics["query_size"] = len(query_idx)

    return metrics


def results_to_dataframe(results_list):
    return pd.DataFrame(results_list)