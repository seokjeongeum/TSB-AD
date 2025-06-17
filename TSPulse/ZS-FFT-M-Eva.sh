PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_fft \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva.csv' \
  --save True
