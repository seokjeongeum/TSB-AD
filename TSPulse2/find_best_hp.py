import pandas as pd
import ast
import numpy as np

def find_best_hp(csv_path):
    """
    Analyzes the hyperparameter tuning results from a CSV file to find the best
    hyperparameter setting.

    The function reads the CSV, calculates the average performance metrics for each
    hyperparameter configuration, and identifies the best one based on VUS-PR
    scores. It fills zero-valued metrics for a file with the values
    from the previous n_components configuration for the same file.

    Args:
        csv_path (str): The path to the hyperparameter tuning results CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Parse the 'HP' column to extract the 'n_components' value
    df['n_components'] = df['HP'].apply(lambda x: ast.literal_eval(x)['n_components'])

    # Identify metric columns
    metric_columns = [col for col in df.columns if col not in ['file', 'HP', 'n_components']]
    
    # --- Before filling ---
    avg_metrics_before = df.groupby('n_components')[metric_columns].mean()
    print("Average metrics for each n_components value:")
    print(avg_metrics_before)

    # Find the best n_components based on 'VUS-PR'
    vus_pr_scores = pd.Series(avg_metrics_before['VUS-PR'])
    best_n_components_vus_pr = vus_pr_scores.idxmax()
    best_vus_pr_score = vus_pr_scores.max()
    print(f"\nBest n_components based on VUS-PR is: {best_n_components_vus_pr}")
    print(f"Average VUS-PR score: {best_vus_pr_score:.4f}")


if __name__ == "__main__":
    csv_file = 'eval/HP_tuning/multi/TSPulse2.csv'
    find_best_hp(csv_file) 