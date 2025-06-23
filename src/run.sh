#!/bin/bash

# Generic script to run chi_square_calculator.py with Hadoop Streaming
# Usage: ./run_chi_square.sh <stopwords_path> <input_file> <output_dir> [hadoop_streaming_jar]

STOPWORDS_PATH=$1
INPUT_FILE=$2
OUTPUT_DIR=$3
HADOOP_STREAMING_JAR=${4:-/usr/lib/hadoop/tools/lib/hadoop-streaming-3.3.6.jar}  # Default if not provided

if [ $# -lt 3 ]; then
  echo "Usage: $0 <stopwords_path> <input_file> <output_dir> [hadoop_streaming_jar]"
  exit 1
fi

python chi_square_calculator.py \
  -r hadoop \
  --hadoop-streaming-jar "$HADOOP_STREAMING_JAR" \
  --stopwords "$STOPWORDS_PATH" \
  "$INPUT_FILE" \
  --output-dir "$OUTPUT_DIR"
