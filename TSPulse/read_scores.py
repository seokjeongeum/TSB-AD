import numpy as np
import os

# Define the directory containing the .npy files
directory_path = "/workspaces/TSB-AD/eval/score/uni/TSPulse_ZS_ensemble/"


def read_and_print_npy_files(path):
    """
    Reads all .npy files in a given directory, sorts them, and prints a summary of their contents.
    """
    # Check if the directory exists
    if not os.path.isdir(path):
        print(f"Error: Directory not found at '{path}'")
        return

    # Get a list of all files in the directory that end with .npy
    try:
        npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return

    if not npy_files:
        print(f"No .npy files found in '{path}'")
        return

    # Sort the files alphabetically to process them in order
    npy_files.sort()

    print(f"Found {len(npy_files)} .npy files. Displaying contents:\n")

    write_csv = []
    # Loop through each sorted .npy file
    for filename in npy_files:
        full_path = os.path.join(path, filename)

        try:
            # Load the numpy array from the file
            output = np.load(full_path)

            # Print a formatted summary
            print(f"--- File: {filename} ---")
            print(f"Shape of array: {output.shape}")

            # Show a preview of the first 5 scores
            preview_scores = output[:5]
            print(f"First 5 scores: {preview_scores}")

            print("-" * (len(filename) + 12))
            print()  # Add a newline for better readability

        except Exception as e:
            print(f"Could not read or process file {filename}: {e}\n")


# Run the function on your specified directory
read_and_print_npy_files(directory_path)
