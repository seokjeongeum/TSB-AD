PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"
export CUDA_VISIBLE_DEVICES="4"
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_ensemble \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_ensemble \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_ensemble \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_ensemble \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'

python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_fft \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_fft \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_fft \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_fft \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'


python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_future \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_future \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_future \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_future \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'

python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_time \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_M.py" \
  --AD_Name=TSPulse_ZS_time \
  --dataset_dir 'Datasets/TSB-AD-M/' \
  --file_lsit 'Datasets/File_List/TSB-AD-M-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/multi-tuning/' \
  --save_dir 'eval/metrics/multi-tuning/'
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_time \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Eva4.csv' \
  --save True
python "${PROJECT_ROOT}/benchmark_exp/Run_Detector_U.py" \
  --AD_Name=TSPulse_ZS_time \
  --dataset_dir 'Datasets/TSB-AD-U/' \
  --file_lsit 'Datasets/File_List/TSB-AD-U-Tuning4.csv' \
  --save True \
  --score_dir 'eval/score/uni-tuning/' \
  --save_dir 'eval/metrics/uni-tuning/'


