# cd external/YOLOX/
# pip install -r requirements.txt && python setup.py develop
# cd external/deep-person-reid/
# pip install -r requirements.txt && python setup.py develop
# cd external/fast_reid/
# pip install -r docs/requirements.txt
#exp = "trackers"
exp = "trackers"
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity \
  --TRACKERS_TO_EVAL /mnt/DATA/jas123/Downloads/OCL_DEEP_OC_SORT/results/trackers/DANCE-val/test5_5_100_post/ \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK DANCE