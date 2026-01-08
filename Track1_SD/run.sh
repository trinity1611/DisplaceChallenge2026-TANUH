#!/bin/bash

CONDA_BASE=$(conda info --base) # or specify the path directly, e.g., ~/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

############### Specifying the data paths 
data_dir_path='data_directory'    			# This will change when running for other tracks
dir_containing_files='data'       			# the data directory. This name will change when running for other tracks
wav_dir='wav'                     			# name of the directory containing the recordings. This name will change to Audio when running for other tracks
rttm_dir='rttm'                   			# name of the directory containing the reference rttms
dest_dir='data_directory/gen_rttm'               	# name of the directory where the DiariZen output rttms need to be stored
config_path='config.toml'         			# Path to the config file

############### Speaker diarization using DiariZen
conda activate diarizen
python3 DiariZen/inference_withConfigFile.py $data_dir_path $dir_containing_files/$wav_dir $dest_dir $config_path 2> inf_log.txt
conda deactivate
echo "Per-file diarization outputs are availabel in $dest_dir"
echo "Please refer to inf_log.txt for errors"

############### Scoring
echo "SCORING ...."
if [ ! -d $data_dir_path/$dir_containing_files/$rttm_dir ]; then
   echo "No Ground Truth rttms provided. Skipping scoring ..." 
else
   if [ ! -d $data_dir_path/score ]; then
      mkdir $data_dir_path/score
   fi
   
   ref_rttm=$data_dir_path/score/ref_all.rttm
   gen_rttm=$data_dir_path/score/gen_all.rttm
   
   cat $data_dir_path/$dir_containing_files/$rttm_dir/* > $ref_rttm
   cat $dest_dir/* > $gen_rttm
   
   DiariZen/dscore/score.py -r $ref_rttm -s $gen_rttm > $data_dir_path/score/final_score.out 2> $data_dir_path/score/final_score.err
   der=$(cat $data_dir_path/score/final_score.out | grep  "*** OVERALL" |awk '{print $4}')
   echo "************************************************"
   echo ""
   echo "              OVERALL DER = $der                "
   echo ""
   echo "************************************************"
   echo "Error and results file present at $data_dir_path/score"
fi
