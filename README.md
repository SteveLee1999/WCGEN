# World-Consistent Data Generation for Vision-and-Language Navigation


## Requirements
1. Create and activate a [conda](https://conda.io/) environment named `wcgen`:

```
conda env create -f environment.yaml
conda activate wcgen
```

2. Install Matterport3D Simulators and prepare the RGB and depth images of Matterport3D following: [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

3. Download R2R datasets from [Dropbox](https://www.dropbox.com/scl/fo/4iaw2ii2z2iupu0yn4tqh/AJutXWSGTtjBFYXnxr-4YQw?rlkey=88khaszmvhybxleyv0a9bulyn&e=1&dl=0)

4. Download our trained models from [GoogleDrive](https://drive.google.com/drive/folders/1zB-XtXPSjSjnmTDRG_QuXkkVY9A4wHtQ?usp=drive_link)
## Running
1. Generate description prompts with BLIP-2:

single GPU:
```
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python 1_caption.py
```
multi GPUs:
```
python dist.py --task "caption" --devices 0 1...n
```


2. Generate panoramas with Stable Diffusion:

single GPU:
```
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python 2_panorama.py
```
multi GPUs:
```
python dist.py --task "panorama" --devices 0 1...n
```


3. Generate instructions with mPLUG:
```
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python 3_instruction.py
```
