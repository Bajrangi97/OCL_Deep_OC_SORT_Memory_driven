# Enhance Deep-OC-SORT : memory Driven






<center>
<img src="pipeline.png" width="600"/>
</center>

Results on  MOT17 Val dataset

| Dataset          | HOTA | AssA | IDF1 | 
| ---------------- | ---- | ---- | ---- | 
| MOT17 | 70.27 | 73.57 | 82.78 |  

Result on DanceTrack Test Dataset

| Dataset          | HOTA | AssA | DetA | MOTA  | IDF1   |
| ---------------- | ---- | ---- | ---- | ---- | ----- | 
| DanceTrack | 62.6 | 48.4 | 81.1 | 89.9| 63.4 | 



## Installation

Tested with Python3.8 on Ubuntu 20.04. More versions will likely work.

After cloning, install external dependencies: 
After cloning the repo, download the cache folder from this link and placed it into Enhance Deep-OC-SORT : memory Driven folder.
cache link: https://drive.google.com/drive/folders/1ucRzQxJJgIez9Bj0PMK0hWcjgAuqDvxp?usp=sharing

```
cd external/YOLOX/
pip install -r requirements.txt && python setup.py develop
cd ../external/deep-person-reid/
pip install -r requirements.txt && python setup.py develop
cd ../external/fast_reid/
pip install -r docs/requirements.txt
```

OCSORT dependencies are included in the external dependencies. If you're unable to install `faiss-gpu` needed by `fast_reid`, 
`faiss-cpu` should be adequate. Check the external READMEs for any installation issues.

Add [the weights](https://drive.google.com/drive/folders/1cCOx_fadIOmeU4XRrHgQ_B5D7tEwJOPx?usp=sharing) to the 
`external/weights` directory (do NOT untar the `.pth.tar` YOLOX files).

## Data

Place MOT17/20 and DanceTrack under:

```
data
|——————mot (this is MOT17)
|        └——————train
|        └——————test
|——————MOT20
|        └——————train
|        └——————test
|——————dancetrack
|        └——————train
|        └——————test
|        └——————val
```
and run:

```
python3 data/tools/convert_mot17_to_coco.py
python3 data/tools/convert_mot20_to_coco.py
python3 data/tools/convert_dance_to_coco.py
```

## Evaluation



For the MOT17/20 and DanceTrack baseline:

```
# exp=baseline
# Flags to disable all the new changes
python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --grid_off --dataset mot17
python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --grid_off -dataset mot20 --track_thresh 0.4
python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --grid_off --dataset dance --aspect_ratio_thresh 1000
```

This will cache detections under ./cache, speeding up future runs. This will create results at:

```
# For the standard results
results/trackers/<DATASET NAME>-val/$exp.
# For the results with post-processing linear interpolation
results/trackers/<DATASET NAME>-val/${exp}_post.
```

To run TrackEval for HOTA and Identity with linear post-processing on MOT17, run:

```bash
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity \
  --TRACKERS_TO_EVAL ${exp}_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK MOT17
```

Replace that last argument with MOT17 / MOT20 / DANCE to evaluate those datasets.  

For the highest reported ablation results, run: 
```
exp=best_paper_ablations
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot17 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot20 --track_thresh 0.4 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset dance --aspect_ratio_thresh 1000 --w_assoc_emb 1.25 --aw_param 1
```

This will cache generated embeddings under ./cache/embeddings, speeding up future runs. Re-run the TrackEval script provided 
above.

You can achieve higher results on individual datasets with different parameters, but we kept them fairly consistent with round 
numbers to avoid over-tuning.

For training the FastReid on Dancetrack dataset:
Go to: ./external/fast_reid/datasets
```
# run the command
python3 generate_mot_patches.py
```
After run this traning data will store at:
OCL_DEEP_OC_SORT-exp/external/fast_reid/datasets/dancetrack-ReID

After That run the command for training the fastReID module (resnet backbone):
Go to ./OCL_DEEP_OC_SORT-exp/external/fast_reid
```
python3 tools/train_net.py --config-file ./configs/dancetrack/bagtricks_R50.yml --num-gpus 4
```
For Training the FastReID on ViT back bone change the yml file 
#./configs/dancetrack/bagtricks_vit.yml
```
python3 tools/train_net.py --config-file ./configs/dancetrack/bagtricks_vit.yml --num-gpus 4
```
After that model will save at:
./external/logs/dancetrack
## Contributing

Formatted with `black --line-length=120 --exclude external .`

# Citation

If you find our work useful, please cite our paper: 
```
{

}

```

Also cite this paper 
```
@article{maggiolino2023deep,
    title={Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification}, 
    author={Maggiolino, Gerard and Ahmad, Adnan and Cao, Jinkun and Kitani, Kris},
    journal={arXiv preprint arXiv:2302.11813},
    year={2023},
}
```

Also see OC-SORT, which we base our work upon: 
```
@article{cao2022observation,
  title={Observation-centric sort: Rethinking sort for robust multi-object tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
