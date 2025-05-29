# Usage
Please refer to [install.md](install.md) for environment setup.</br>
Please refer to [waymo.md](waymo.md) for preparing dataset.

## Overview
The pipeline of this project consists of two main modules:  
1. Training the 3DGS framework for urban scene reconstruction and simulation
2. Evaluating the Learned 3DGS for Reconstruction and Editing.

### Running
``` python
cd ./drivestudio</br>
export PYTHONPATH=$(pwd)
# training
python tools/train.py --config_file configs/omnire.yaml --run_name idx dataset=waymo/1cams data.scene_idx=idx  data.start_timestep=0 data.end_timestep=-1

# test(rebuild)
python3 tools/eval.py --resume_from ./work_dirs/drivestudio/idx/checkpoint_final.pth

# test(delete idx instance)
python3 tools/eval.py --delete_ids idx --render_video_postfix delete_3dgs  --resume_from ./work_dirs/drivestudio/idx/checkpoint_final.pth 
# simulate result will be find in ./work_dirs/drivestudio/idx/video_eval/
```
