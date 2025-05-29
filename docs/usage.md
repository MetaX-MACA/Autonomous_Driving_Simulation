# Usage
Please refer to [install.md](install.md) for environment setup.</br>

## Overview
The pipeline of this project consists of three main modules:  
1. Use 3DGS for scene reconstruction.
2. Use LLM for scene edit.
3. Use SD for scene transition.

### 1. SceneReconstruction
Please refer to [usage.md](../SceneReconstruction/docs/usage.md) for environment setup.

## 2. SceneEdit
Please refer to [usage.md](../SceneEdit/docs/usage.md) for environment setup.

## 3. SceneTransition
Please refer to [usage.md](../SceneTransition/docs/usage.md) for environment setup.


## Note
If you want to edit a specific scene for data augmentation, the reconstruction and simulation results from SceneReconstruction can be used as input for SceneEdit. Once the editing process is complete, the output from SceneEdit can then be passed to SceneTransition.By following this workflow, you can generate diverse and enriched data based on a single scene, effectively enhancing datasets for autonomous driving tasks.
``` shell
# Decompress the output .mp4 file from SceneReconstruction, and update the init_img_dir field in the SceneEdit YAML configuration file accordingly.
init_img_dir: /path/to/your/scenereconstruction/render_png

# Use the output from SceneEdit as the input for SceneTransition by specifying it as the video_fold
--video_fold /path/to/your/sceneedit/videos
```

