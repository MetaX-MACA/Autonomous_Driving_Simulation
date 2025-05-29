# Installation

- [Installation](#installation)
  - [Install MetaX C500 Driver and SDK](#install-metax-c500-driver-and-sdk)
  - [Install ChatSim runtime environment](#install-chatsim-runtime-environment)
    - [Clone Chatsim repo recursively](#clone-chatsim-repo-recursively)
    - [Replace files](#replace-files)
      - [What we do on ChatSim](#what-we-do-on-chatsim)
    - [Step1:Create python environment](#step1create-python-environment)
      - [Step1.1:Using Conda to create](#step11using-conda-to-create)
      - [Step1.2:Install PyTorch and vLLM from MetaX Developer Center](#step12install-pytorch-and-vllm-from-metax-developer-center)
      - [Step1.3:Install python packages](#step13install-python-packages)
    - [Step2: Install Blender Software and Blender Utils](#step2-install-blender-software-and-blender-utils)
    - [Step3: Install background rendering engine (optional)](#step3-install-background-rendering-engine-optional)
    - [Step4: Install Inpainting tools (optional)](#step4-install-inpainting-tools-optional)
    - [Step5: Setup Trajectory Tracking Module (optional)](#step5-setup-trajectory-tracking-module-optional)
    - [Step6: Install McLight (optional)](#step6-install-mclight-optional)


Recommended runtime environment:
- OS: Ubuntu >= 22.04
- Python: >= 3.10
- PyTorch: >= 2.4
- GPU: MetaX C500
- Driver/SDK Version: 2.31.0.4 or higger
- RAM: >= 128GB
- LLM API Key which support OpenAI SDK

You can get free LLM API Token on [GitHub](https://github.com/marketplace?type=models), but it has [rate limits](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits), for [OpenAI GPT-4.1-mini](https://github.com/marketplace/models/azure-openai/gpt-4-1-mini), 	requests per minute	is 15，requests per day is 150.

**You can alse deploy the LLM model service locally on our MetaX C500 GPU! This will eliminate any rate limitations!**

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

## Install ChatSim runtime environment

Install dependencies
```shell
apt-get update
apt-get install libxrender1 libxi6 libxkbcommon0 libsm6 libice6
```
### Clone Chatsim repo recursively
First clone Chatsim repo recursively.
``` shell
git clone https://github.com/yifanlu0227/ChatSim  --recursive
```

### Replace files
Execute the following script, using the files in the `metax_modifications` directory, to copy and replace the corresponding files in the original Chatsim.
```shell
bash replace_files.sh
```
#### What we do on ChatSim
In order to build the MetaX-Autonomous Driving Scene Reconstruction & Simulation Solution, we have made the following modifications to the `ChatSim` project to adapt to the new process:

- Add more LLM agents api that follow openai SDK
- The generation method of chatsim background image is adjusted, and it is changed to generate background image by other modules in the solution. Chatsim only needs to load and use the generated image. Similarly, when generating a new image, it is no longer necessary to modify the camera external parameters
- When rendering scenes with blender, the CPU mode is used instead, because the current MetaX GPU does not support image rendering
- Solve the problem of unable to generate a reverse vehicle sample bug
- Some super parameters were adjusted when generating samples

### Step1:Create python environment
#### Step1.1:Using Conda to create
``` shell
conda create -n chatsim python=3.8
conda activate chatsim
```

#### Step1.2:Install PyTorch and vLLM from MetaX Developer Center
Download PyTorch from MetaX Developer Center [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)

**Note** Please download the version that matches the Driver, such as `2.31.x.x`.

You will receive a tar archives. After extracting it, navigate to the `wheel` directory and install using `pip`.
``` shell
tar -xvf maca-pytorch2.0-py38-2.31.0.4-x86_64.tar.xz
cd 2.31.0.4/wheel/
pip install ./*.whl
```

If you want to deploy the LLM model service locally, you also need to download and install vLLM package. [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm&ai_label=vLLM)
```
# install vLLM
tar -xvf mxc500-vllm-py310-2.31.0.4-linux-x86_64.tar.xz
cd mxc500-vllm-2.31.0.4/wheel/
pip install ./*.whl
```

#### Step1.3:Install python packages
``` shell
cd ./ChatSim
pip install -r requirements.txt
```
### Step2: Install Blender Software and Blender Utils
Please refer to the installation steps provided by the original author, [link](https://github.com/yifanlu0227/ChatSim?tab=readme-ov-file#step-4-install-blender-software-and-our-blender-utils)

**Note**: To use Blender, you need to add Blender's Python path to the environment variables, as shown below:
```shell
cd chatsim/foreground/Blender
export blender_py=$PWD/blender-3.5.1-linux-x64/3.5/python/bin/python3.10
```
To test ChatSim each time, you need to set this environment variable. You can add it to the `~/.bashrc` file.

### Step3: Install background rendering engine (optional)
Since the MetaX version uses images rendered by other modules, this step can be skipped.

If you want to use, please refer to the installation steps provided by the original author, [link](https://github.com/yifanlu0227/ChatSim?tab=readme-ov-file#step-2-install-background-rendering-engine)

**Note**: During the installation process, replace the PyTorch-related libraries with the MetaX version and resolve any dependency version conflicts.

### Step4: Install Inpainting tools (optional)
This module is used for editing background images, such as removing all background vehicles.

If you want to use, please refer to the installation steps provided by the original author, [link](https://github.com/yifanlu0227/ChatSim?tab=readme-ov-file#step-3-install-inpainting-tools)

**Note**: During the installation process, replace the PyTorch-related libraries with the MetaX version and resolve any dependency version conflicts.

### Step5: Setup Trajectory Tracking Module (optional)
This module is used to smooth the motion trajectory of inserted vehicles.

If you want to use, please refer to the installation steps provided by the original author, [link](https://github.com/yifanlu0227/ChatSim#step-5-setup-trajectory-tracking-module-optional)

### Step6: Install McLight (optional)
This module is used to train the `skydome` model and infer the `skydome` in new scenes.

If you want to use, please refer to the installation steps provided by the original author, [link](https://github.com/yifanlu0227/ChatSim#step-6-install-mclight-optional)

**Note**: 
- If you want to obtain the `skydome` for a new scene, you need to install this module and download the pre-trained model provided by the author.  
- If you prefer not to install it, you can temporarily use the `skydome` files from other scenes provided by the author as a substitute. For specific instructions, refer to the usage.md.
- When performing inference, please refer to this [issue](https://github.com/yifanlu0227/ChatSim/issues/48) to configure the settings file.
