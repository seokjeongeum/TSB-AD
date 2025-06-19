PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"
export CUDA_VISIBLE_DEVICES="7"
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse2 \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva7.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse2 \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva7.csv' \
  --save True
