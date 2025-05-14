#!/bin/bash

PYTHON_EXE="python3 -m"
MAIN_SCRIPT="benchmark_exp.Run_Custom_Detector"
MAIN_FILE_LIST="Datasets/File_List/TSB-AD-M-Eva.csv"
TEMP_FILE_LIST_DIR="temp_file_lists"
SUPERVISION_MODE="supervised"
PART_NUM=1
TOTAL_PARTS=3

mkdir -p "${TEMP_FILE_LIST_DIR}"
TEMP_FILE_LIST="${TEMP_FILE_LIST_DIR}/file_list_part_${PART_NUM}_${SUPERVISION_MODE}.csv"

# Check if main file list exists
if [ ! -f "${MAIN_FILE_LIST}" ]; then
    echo "ERROR: Main file list ${MAIN_FILE_LIST} not found."
    exit 1
fi

# Get header
HEADER=$(head -n 1 "${MAIN_FILE_LIST}")
# Get total number of data files (excluding header)
TOTAL_FILES=$(tail -n +2 "${MAIN_FILE_LIST}" | wc -l)
if [ "${TOTAL_FILES}" -eq 0 ]; then
    echo "ERROR: No files found in ${MAIN_FILE_LIST} (excluding header)."
    exit 1
fi

CHUNK_SIZE=$(( (TOTAL_FILES + TOTAL_PARTS - 1) / TOTAL_PARTS )) # Ceiling division
START_LINE=$(( (PART_NUM - 1) * CHUNK_SIZE + 2 )) # +2 because tail is 1-indexed and we skip header
END_LINE=$(( START_LINE + CHUNK_SIZE - 1 ))

echo "Processing Part ${PART_NUM}/${TOTAL_PARTS} in ${SUPERVISION_MODE} mode."
echo "Total files: ${TOTAL_FILES}, Chunk size: ${CHUNK_SIZE}"
echo "Reading lines from ${START_LINE} to ${END_LINE} of ${MAIN_FILE_LIST}"

# Create temporary file list for this part
echo "${HEADER}" > "${TEMP_FILE_LIST}"
tail -n +${START_LINE} "${MAIN_FILE_LIST}" | head -n ${CHUNK_SIZE} >> "${TEMP_FILE_LIST}"

# Verify temp file list
TEMP_COUNT=$(tail -n +2 "${TEMP_FILE_LIST}" | wc -l)
echo "Temporary file list ${TEMP_FILE_LIST} created with ${TEMP_COUNT} files."

# Run the Python script
echo "Running: ${PYTHON_EXE} ${MAIN_SCRIPT} --file_list_path ${TEMP_FILE_LIST}"
"${PYTHON_EXE}" "${MAIN_SCRIPT}" --file_list_path "${TEMP_FILE_LIST}"

# Clean up
echo "Cleaning up temporary file list: ${TEMP_FILE_LIST}"
rm "${TEMP_FILE_LIST}"
echo "Part ${PART_NUM} (${SUPERVISION_MODE}) finished." 