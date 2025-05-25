import csv
import io

# table2_data_str will be populated by the platform or defined here
table2_data_str = """
# TSB-AD-U
# Name, #TS Collected, #TS Curated, Avg Dim, Avg TS Len, Avg # Anomaly, Avg Anomaly Len, Anomaly Ratio, Category
UCR [106], 250, 228, 1, 67818.7, 1, 198.9, 0.6%, P&Seq
NAB [6], 58, 28, 1, 5099.7, 1.6, 370.1, 10.6%, Seq
YAHOO [51], 367, 259, 1, 1560.2, 5.5, 2.5, 0.6%, P&Seq
IOPS [1], 58, 17, 1, 72792.3, 25.6, 48.7, 1.3%, Seq
MGAB [100], 10, 9, 1, 97777.8, 9.7, 20.0, 0.2%, Seq
WSD [113], 210, 111, 1, 17444.5, 5.1, 25.4, 0.6%, Seq
SED [20], 6, 3, 1, 23332.3, 14.7, 64.0, 4.1%, Seq
TODS [50], 15, 15, 1, 5000.0, 97.3, 18.7, 6.3%, P&Seq
NEK [96], 48, 9, 1, 1073.0, 2.9, 51.1, 8.0%, P&Seq
Stock [101], 90, 20, 1, 15000.0, 1246.9, 1.1, 9.4%, P&Seq
Power [47], 1, 1, 1, 35040.0, 4, 750, 8.5%, Seq
Daphnet (U) [10], -, 1, 1, 38774.0, 6, 384.3, 5.9%, Seq
CATSv2 (U) [30], -, 1, 1, 300000.0, 19.0, 778.9, 4.9%, Seq
SWaT (U) [62], -, 1, 1, 419919.0, 27.0, 1876.0, 12.1%, Seq
LTDB (U) [35], -, 9, 1, 99700.0, 127.5, 144.5, 18.6%, Seq
TAO (U) [2], -, 3, 1, 10000.0, 838.7, 1.1, 9.4%, P&Seq
Exathlon (U) [44], -, 32, 1, 44075.8, 3.1, 1577.3, 11.0%, Seq
MITDB (U) [35], -, 8, 1, 631250.0, 68.7, 451.9, 4.2%, Seq
MSL (U) [43], -, 9, 1, 3492.0, 1.3, 130.0, 5.8%, Seq
SMAP (U) [43], -, 19, 1, 7700.2, 1.2, 210.1, 2.8%, Seq
SMD (U) [97], -, 38, 1, 24207.7, 2.4, 173.7, 2.0%, Seq
SVDB (U) [39], -, 20, 1, 171380.0, 36.4, 292.5, 3.6%, Seq
OPP (U) [88], -, 29, 1, 16544.8, 1.4, 653.4, 6.4%, Seq
# TSB-AD-M
# Name, #TS Collected, #TS Curated, Avg Dim, Avg TS Len, Avg # Anomaly, Avg Anomaly Len, Anomaly Ratio, Category
GHL [29], 48, 25, 19, 199001.0, 2.2, 1035.2, 1.1%, Seq
Daphnet [10], 17, 1, 9, 38774.0, 6.0, 384.3, 5.9%, Seq
Exathlon [44], 72, 27, 21, 60878.4, 4.3, 1373.3, 9.8%, Seq
Genesis [103], 1, 1, 18, 16220.0, 3.0, 16.7, 0.3%, Seq
OPP [88], 24, 8, 248, 17426.75, 1.4, 394.3, 4.1%, Seq
SMD [97], 28, 22, 38, 25466.4, 8.9, 112.8, 3.8%, Seq
SWaT [62], 4, 2, 59, 207457.5, 16.5, 1093.6, 12.7%, Seq
PSM [3], 1, 1, 25, 217624.0, 72.0, 338.6, 11.2%, P&Seq
SMAP [43], 54, 27, 25, 7855.9, 1.3, 196.3, 2.9%, Seq
MSL [43], 27, 16, 55, 3119.4, 1.3, 111.7, 5.1%, Seq
CreditCard [95], 1, 1, 29, 284807.0, 465.0, 1.1, 0.2%, P&Seq
GECCO [64], 1, 1, 9, 138521.0, 51.0, 33.8, 1.2%, Seq
MITDB [35], 48, 13, 2, 336153.8, 15.2, 1846.8, 2.7%, Seq
SVDB [39], 78, 31, 2, 207122.6, 68.3, 268.2, 4.8%, Seq
LTDB [35], 7, 5, 2, 100000.0, 105.0, 134.4, 15.5%, Seq
CATSv2 [30], 10, 6, 17, 240000.0, 11.5, 811.6, 3.7%, Seq
TAO [2], 45, 13, 3, 10000.0, 788.2, 1.1, 8.7%, P&Seq
"""


# Helper function to parse Table 2
def parse_table2_data_internal(table2_str_content):
    u_dims = {}
    m_dims = {}
    current_section = None
    lines = table2_str_content.strip().split("\n")

    start_parsing = False
    for i, line_check in enumerate(lines):
        if (
            line_check.strip().startswith("# Name,")
            or line_check.strip().startswith("Name,")
            or line_check.strip().startswith("# TSB-AD-U")
            or line_check.strip().startswith("# TSB-AD-M")
        ):
            lines = lines[i:]  # Start parsing from this line
            start_parsing = True
            break
    if not start_parsing:
        return u_dims, m_dims

    for line in lines:
        line = line.strip()
        if line.startswith("# TSB-AD-U"):
            current_section = "U"
            # After identifying a section, skip its own header line in the next iteration
            if lines.index(line) + 1 < len(lines) and (
                lines[lines.index(line) + 1].strip().startswith("# Name,")
                or lines[lines.index(line) + 1].strip().startswith("Name,")
            ):
                continue  # Skip explicit header line if it immediately follows section marker
            continue
        if line.startswith("# TSB-AD-M"):
            current_section = "M"
            if lines.index(line) + 1 < len(lines) and (
                lines[lines.index(line) + 1].strip().startswith("# Name,")
                or lines[lines.index(line) + 1].strip().startswith("Name,")
            ):
                continue
            continue

        # Skip any other headers or irrelevant lines
        if not line or line.startswith("# Name,") or line.startswith("Name,"):
            continue

        parts = [p.strip() for p in line.split(",")]
        # Ensure we are looking at a data row with enough parts
        if (
            len(parts) < 4 or not parts[3].replace(".", "", 1).isdigit()
        ):  # Check if 4th part is likely a number
            if not (
                len(parts) >= 2 and parts[1] == "-"
            ):  # Allow for '-' in #TS Collected / Curated
                continue

        raw_name_field = parts[0]
        try:
            # Avg Dim is expected at index 3 (4th column)
            avg_dim_str = parts[3]
            avg_dim = int(avg_dim_str)
        except (ValueError, IndexError):
            # print(f"Debug: Could not parse Avg Dim for line: '{line}'")
            continue

        base_name = raw_name_field.split(" [")[0]

        if current_section == "U":
            normalized_base_name = base_name.replace(" (U)", "")
            u_dims[normalized_base_name] = avg_dim
        elif current_section == "M":
            m_dims[base_name] = avg_dim

    return u_dims, m_dims


# Helper function to get Avg Dim for a file's base name
def get_avg_dim_for_file_internal(base_name, u_dims_map, m_dims_map):
    if base_name in u_dims_map:
        return u_dims_map[base_name]
    if base_name in m_dims_map:
        return m_dims_map[base_name]
    return float("inf")


# Helper function to extract base dataset name from filename
def get_base_dataset_name_internal(f_name_str):
    name_part = f_name_str
    if name_part.endswith(".csv"):
        name_part = name_part[:-4]

    parts = name_part.split("_")
    if len(parts) > 1 and parts[0].isdigit():
        if parts[1].lower() == "creditcard":
            return "CreditCard"
        return parts[1]
    return parts[0]


def sort_files_by_criteria_from_paths_and_string(
    file_list_path, csv_data_path, table2_content_as_string
):
    try:
        with open(file_list_path, "r", encoding="utf-8") as f:
            files_to_sort_list_str = f.read()
        with open(csv_data_path, "r", encoding="utf-8") as f:
            data_csv_str = f.read()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return []
    except Exception as e:
        print(f"Error reading files: {e}")
        return []

    files_to_sort = files_to_sort_list_str.strip().split("\n")
    if files_to_sort and files_to_sort[0].strip().lower() == "file_name":
        files_to_sort = files_to_sort[1:]
    files_to_sort = [f.strip() for f in files_to_sort if f.strip()]

    key1_values_map = {}
    score_cols_for_avg = [
        "IForest",
        "LOF",
        "PCA",
        "HBOS",
        "OCSVM",
        "MCD",
        "KNN",
        "KMeansAD",
        "COPOD",
        "CBLOF",
        "EIF",
        "RobustPCA",
        "AutoEncoder",
        "CNN",
        "LSTMAD",
        "TranAD",
        "AnomalyTransformer",
        "OmniAnomaly",
        "USAD",
        "Donut",
        "TimesNet",
        "FITS",
        "OFA",
    ]

    csv_file_obj = io.StringIO(data_csv_str)
    csv_dict_reader = csv.DictReader(csv_file_obj)
    for r in csv_dict_reader:
        fname_csv = r.get("file") or r.get("file_name")
        if not fname_csv:
            continue

        current_scores = []
        for col_name in score_cols_for_avg:
            if col_name in r:
                try:
                    current_scores.append(float(r[col_name]))
                except (ValueError, TypeError):
                    pass

        if current_scores:
            key1_values_map[fname_csv] = sum(current_scores) / len(current_scores)
        else:
            key1_values_map[fname_csv] = float("inf")

    # Use the passed string for Table 2 data
    u_dimensions, m_dimensions = parse_table2_data_internal(table2_content_as_string)

    files_data_for_sorting = []
    for file_n in files_to_sort:
        avg_score_key1 = key1_values_map.get(file_n, float("inf"))

        base_ds_name = get_base_dataset_name_internal(file_n)
        avg_dim_key2 = get_avg_dim_for_file_internal(
            base_ds_name, u_dimensions, m_dimensions
        )

        files_data_for_sorting.append((file_n, avg_score_key1, avg_dim_key2))

    files_data_for_sorting.sort(key=lambda item: (item[1], item[2], item[0]))

    result_sorted_filenames = [data_item[0] for data_item in files_data_for_sorting]
    return result_sorted_filenames


def solve():
    global table2_data_str  # Ensure this global string is used

    path_to_file_list = (
        "Datasets/File_List/TSB-AD-M-Eva-Debug.csv"  # Or whatever the platform names it
    )
    path_to_csv_data = "benchmark_exp/benchmark_eval_results/multi_mergedTable_VUS-PR.csv"  # Or whatever the platform names it

    # Call the function that reads two files from paths and one from string
    sorted_list = sort_files_by_criteria_from_paths_and_string(
        path_to_file_list,
        path_to_csv_data,
        table2_data_str,  # Pass the global string here
    )
    return sorted_list


# Example of how you might run this locally for testing:
if __name__ == "__main__":
    print("Running local test with Table 2 as a string...")
    sorted_filenames = solve()
    print("\nSorted file names:")
    for name in sorted_filenames:
        print(name)
