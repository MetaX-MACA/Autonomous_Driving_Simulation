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
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
    export PATH=${MACA_PATH}/bin:${MACA_PATH}/tools/cu-bridge/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_CLANG_PATH}:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
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
conda create -n instv2v python=3.10
conda activate instv2v
```

## Additional required documents
Follow these instructions to manipulate the file.
``` shell
git clone https://github.com/amazon-science/instruct-video-to-video

# Copy the required files
cp ./use_files/infer_base.py ./instruct-video-to-video/
cp ./use_files/requirements_instv2v.txt ./instruct-video-to-video/requirements.txt
```
Comment the following codes in ./instruct-video-to-video/pl_trainer/instruct_p2p_video.py (lines 27, 28) 

```
# self.unet.enable_xformers_memory_efficient_attention()
# self.unet.enable_gradient_checkpointing()
```


### Install python packages
``` shell
cd ./instruct-video-to-video
pip install -r requirements.txt
```

You will receive two tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.4-py310-2.31.0.4-x86_64.tar.xz
cd 2.31.0.4/wheel/
pip install ./*.whl
```

## Infer
Download pretrain checkpoint [insv2v.pth](https://drive.google.com/file/d/166xbfzzxGQgJkbSfzs9nzddzmi_Uh_Kz/view) and run follow cmd.
``` shell
python infer_base.py \
--video_fold /path/to/your/videos \     # input video Path
--prompt "make sunny sky to foggy." \     # edit prompt
--save_fold /path/to/save/results \     # edited video save Path
--image-size 640 640 \            # num frames
--ckpt-path /path/to/your/checkpoint    # instv2v.pth weight file Path
```
