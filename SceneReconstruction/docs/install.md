# Installation

Recommended runtime environment:

OS: Ubuntu 22.04

Python: 3.10

PyTorch >= 2.0

GPU: MetaX C500

Driver/SDK Version: 2.31.0.4 or higger

RAM: >= 128GB

## Install MetaX C500 Driver and SDK
1. Go to the [MetaX Developer Center](https://sw-developer.metax-tech.com/member.php?mod=register) to register an account.

2. Download MetaX C500 [Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85) and [SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85), version: 2.31.0.4 or higger. Please download the local install version.

3. Follow the instructions on the webpage to complete the installation.

4. Update `.bashrc` file.
    
    ```
    vi ~/.bashrc
    ```
    Add the following environment variables
    ```shell
    export MACA_PATH=/opt/maca
    export CUCC_PATH=/opt/maca/tools/cu-bridge
    export CUDA_PATH=$CUCC_PATH
    export PATH=${CUDA_PATH}/bin:${CUCC_PATH}/tools:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/lib:${LD_LIBRARY_PATH}

    export MXLOG_LEVEL=err
    ```
    Update environment variables
    ```
    source ~/.bashrc
    ```
5. Add current user to video group.

    ```shell
    sudo usermod -aG video $USER
    newgrp video
    ```

6. Reboot your server.

7. You can use the `mx-smi` command to check GPU information.

## Using Conda(Suggested)
### Create python environment
``` shell
# create conda environment
conda create -n drivestudio python=3.10
conda activate drivestudio
```

### Download PyTorch and vLLM from MetaX Developer Center
**Note** Please download the version that matches the Driver, such as `2.32.x.x`.

PyTorch: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)


You will receive two tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.0-py310-2.32.0.3-x86_64.tar.xz
cd 2.32.0.3/wheel/
pip install ./*.whl
```

### Install python packages
``` shell
cd ./opensourcedrivestudio
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7

git clone https://github.com/ziyc/drivestudio.git
cd drivestudio
# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../../..
```

## Additional required documents
Since the current project leverages the multi-objective optimization-based reinforcement learning algorithm from Stage-Wise-CMORL, please follow these steps:
``` shell
# 1. Copy the required files
cp ./models/nodes/rigid.py ./drivestudio/models/nodes/
cp ./models/trainers/base.py ./drivestudio/models/trainers/
cp ./models/modules.py ./drivestudio/models/
cp ./models/gaussians/vanilla.py ./drivestudio/models/gaussians/
cp ./tools/eval.py ./drivestudio/tools/
cp ./tools/train.py ./drivestudio/tools/
cp ./configs/omnire.yaml ./drivestudio/configs/
cp ./datasets/tools/extract_masks.py ./drivestudio/datasets/tools/
```
