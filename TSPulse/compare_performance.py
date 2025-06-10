import pandas as pd
import os

def process_and_print_comparison(filepath: str, title: str):
    """
    Reads a CSV file with benchmark results, compares TSPulse (FT)
    against the best other algorithm for each dataset, sorts by improvement,
    and prints the results.

    Args:
        filepath (str): The path to the CSV file.
        title (str): The title for the output section (e.g., "Univariate").
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    try:
        df = pd.read_csv(filepath)
        df.set_index('Method', inplace=True)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error processing file {filepath}: {e}")
        return

    tspulse_ft_scores = df.loc['TSPulse (FT)']
    other_algos_df = df.drop(index=['TSPulse (FT)', 'TSPulse (ZS)'])

    best_other_scores = other_algos_df.max()
    best_other_algos = other_algos_df.idxmax()

    # --- Store results in a list of dictionaries for sorting ---
    results_list = []
    for dataset in df.columns:
        tspulse_score = tspulse_ft_scores[dataset]
        best_algo_name = best_other_algos[dataset]
        best_algo_score = best_other_scores[dataset]

        if best_algo_score > 0:
            improvement = ((tspulse_score - best_algo_score) / best_algo_score) * 100
        else:
            # Handle cases where best score is 0; avoid division by zero
            # If TSPulse score is also 0, improvement is 0. If > 0, it's infinite.
            # We can represent 'infinite' with a very large number for sorting purposes.
            improvement = float('inf') if tspulse_score > 0 else 0.0

        results_list.append({
            "dataset": dataset,
            "tspulse_score": tspulse_score,
            "best_algo_name": best_algo_name,
            "best_algo_score": best_algo_score,
            "improvement": improvement
        })

    # --- Sort the list by the 'improvement' key in descending order ---
    sorted_results = sorted(results_list, key=lambda x: x['improvement'], reverse=True)

    # --- Print the formatted output ---
    print("-" * 80)
    print(f"Comparison for {title} Anomaly Detection (VUS-PR Score)")
    print("(Sorted by Improvement %)")
    print("-" * 80)
    header = f"{'Dataset':<15} | {'TSPulse (FT)':<12} | {'Best Algorithm':<22} | {'Best Score':<12} | {'Improvement'}"
    print(header)
    print("-" * 80)

    for result in sorted_results:
        # Format the improvement string for display
        if result['improvement'] == float('inf'):
             improvement_str = "Inf"
        else:
             improvement_str = f"{result['improvement']:>7.2f}%"

        row_str = (
            f"{result['dataset']:<15} | "
            f"{result['tspulse_score']:<12.3f} | "
            f"{result['best_algo_name']:<22} | "
            f"{result['best_algo_score']:<12.3f} | "
            f"{improvement_str}"
        )
        print(row_str)

    print("-" * 80)
    print("\n")


def main():
    """
    Main function to run the performance comparison for both
    univariate and multivariate datasets.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uni_csv_path = os.path.join(script_dir, "uni.csv")
    multi_csv_path = os.path.join(script_dir, "multi.csv")

    process_and_print_comparison(
        filepath=uni_csv_path,
        title="Univariate"
    )

    process_and_print_comparison(
        filepath=multi_csv_path,
        title="Multivariate"
    )

if __name__ == "__main__":
    main()