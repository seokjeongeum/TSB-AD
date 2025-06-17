PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_time \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva.csv' \
  --save True
