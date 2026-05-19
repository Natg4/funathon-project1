#!/bin/bash
FILE_1="intermediate_solutions/1_preprocessing.py"
mkdir -p $(dirname "$FILE_1")
FILE_2A="intermediate_solutions/2a_GB.py"
mkdir -p $(dirname "$FILE_2A")
FILE_2B="intermediate_solutions/2b_RF.py"
mkdir -p $(dirname "$FILE_2B")
FILE_3="intermediate_solutions/3_metrics.py"
mkdir -p $(dirname "$FILE_3")

# Step 1 - preprocessing
bash solution/admin/extract.sh "subject/1-preprocessing.qmd" $FILE_1

# File step2a - GB
bash solution/admin/extract.sh "subject/2-GB_model.qmd" $FILE_2A

# File step2b - RF
bash solution/admin/extract.sh "subject/2-RF_model.qmd" $FILE_2B

# File step3 - metrics
bash solution/admin/extract.sh "subject/3-metrics.qmd" "temp.py" && cat intermediate_solutions/0_fallback.py temp.py > $FILE_3 && rm temp.py
