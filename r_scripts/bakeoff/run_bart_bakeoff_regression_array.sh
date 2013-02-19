#!/bin/bash

#$ -j y
#$ -N tf_bart_bakeoff
#$ -t 1-100

echo "starting R bart_bakeoff for iteration number # $SGE_TASK_ID"
export _JAVA_OPTIONS="-Xms128m -Xmx5300m"

R --no-save --args iter_num=$SGE_TASK_ID < r_scripts/bakeoff/bart_bakeoff_regression.R