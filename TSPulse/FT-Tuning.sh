PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_FT_ensemble \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_FT_time \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_FT_fft \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_FT_future \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_FT_ensemble \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_FT_time \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_FT_fft \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_FT_future \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'
