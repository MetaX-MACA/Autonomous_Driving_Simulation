# Usage

- [Usage](#usage)
  - [Data Preparation](#data-preparation)
    - [Download and extract Waymo data](#download-and-extract-waymo-data)
    - [Download recalibrate Waymo data (optional)](#download-recalibrate-waymo-data-optional)
    - [Download Blender 3D Assets](#download-blender-3d-assets)
    - [Download Skydome HDRI](#download-skydome-hdri)
  - [Train](#train)
  - [Simulation](#simulation)
    - [Modify configuration](#modify-configuration)
    - [Set LLM API Key](#set-llm-api-key)
    - [LLM model preparation (optional)](#llm-model-preparation-optional)
    - [Start simulation](#start-simulation)

**Note** 
- Please refer to [install.md](install.md) for environment setup.
- All the following operations should be performed within the ChatSim directory after git cloning.

## Data Preparation

### Download and extract Waymo data
Please refer to the waymo download steps provided by the original author. [link](https://github.com/yifanlu0227/ChatSim#data-preparation)

When downloading data, navigate to the [training](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_2/individual_files/training) and [validation](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_2/individual_files/validation) directories separately, and download the multiple files specified by the original author.


**Note** 
- It is recommended to use the `Filter` on the page to select the required files, **and download them directly via the webpage or a download tool**, as using gcloud in China may encounter network connectivity issues.

- please download the following files in the [training](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_2/individual_files/training) directory to demonstrate the MetaX autonomous driving solution.

  ```
  segment-9653249092275997647_980_000_1000_000_with_camera_labels.tfrecord
  ```

### Download recalibrate Waymo data (optional)
Since the built-in NeRF or 3DGS modules are not used, this step can be skipped.

Please refer to the recalibrate waymo data download steps provided by the original author.[link](https://github.com/yifanlu0227/ChatSim#recalibrate-waymo-data)

**Note** 
- The additional Waymo data downloaded for this solution cannot be processed using the original ChatSim's NeRF or 3DGS modules.
- If you want to use these modules, you need to additionally install and compile the COLMAP software, then preprocess the Waymo data. Processing a single data entry may take over 20 hours.

### Download Blender 3D Assets
This asset includes a variety of 3D vehicle models for inserting vehicles into the scene. Please refer to the download steps provided by the original author.[link](https://github.com/yifanlu0227/ChatSim#download-blender-3d-assets)

### Download Skydome HDRI
This file represents the environmental lighting intensity in the scene and is used in Blender to render appropriate shadow effects for inserted vehicles.Please refer to the download steps provided by the original author.[link](https://github.com/yifanlu0227/ChatSim#download-skydome-hdri)

For the additional Waymo data downloaded in this solution, there are two ways to obtain the Skydome HDRI file:
1. Refer to the [install.md](./install.md) to additionally install the McLight module, download the pre-trained model provided by the author, and perform inference to generate the Skydome HDRI file.
2. For a quick experience of the ChatSim workflow, you can temporarily use the Skydome HDRI files generated by the original author for other datasets by simply modifying the directory name. For example, copy and use the Skydome HDRI file from `segment-11379226583756500423_6230_810_6250_810_with_camera_labels`.
 
## Train
To experience the background modeling module training (NeRF or 3DGS) in the original ChatSim, please refer to the steps provided by the original author. [link](https://github.com/yifanlu0227/ChatSim#train-and-simulation)

## Simulation
### Modify configuration
Since the MetaX autonomous driving solution has adjusted the background image generation method in ChatSim, switching to using pre-generated background images produced by the solution's front-end modules, ChatSim now only needs to load and utilize these pre-generated images.

As a result, modifications are required to the original configuration files in ChatSim. For detailed explanations of each parameter in the original configuration files, please refer to [link](https://github.com/yifanlu0227/ChatSim#config-file-explanation)

Modifications are as follow:

Taking the `config/3dgs-waymo-9653.yaml` file as an example, add the `init_img_dir` parameter below the `pcd_file` parameter to specify the directory name of the background images generated by the front-end module. A sample is shown below:
```yaml
scene_name: &scene_name segment-9653249092275997647_980_000_1000_000_with_camera_labels

scene:
  data_root: data/waymo_multi_view
  scene_name: *scene_name
  
  # suppose these files are under $data_root/$scene_name/ 
  ext_int_file: 'cams_meta_waymo.npy'       # extrinsic and intrinsic parameters
  bbox_file: '3d_boxes.npy'
  map_file: 'map.pkl'
  init_img_file: 'wide_init_img.png'
  pcd_file: 'point_cloud/000_TOP.ply'
  init_img_dir: 'render_png'
```

The `init_img_dir` directory is located under the preprocessed Waymo data directory. An example is shown below:
```
data
`-- waymo_multi_view
    |-- ...
    `-- segment-9653249092275997647_980_000_1000_000_with_camera_labels
        |-- ...
        |-- render_png 
```

**Note** 

The MetaX autonomous driving solution currently **only supports front-view** video generation and does not support wide mode.

If background images generated by the preprocessing module are not yet available, you may use images from the original dataset as a substitute to test the ChatSim module independently. The usage method is as follows:
1. Modify the `waymo_data_dir` parameter in the `data_utils/create_front_img.py` file to specify the absolute path of the preprocessed Waymo data directory, such as:
   ```python
    waymo_data_dir = "/home/metax/metax-ChatSim/ChatSim/data/waymo_multi_view/segment-9653249092275997647_980_000_1000_000_with_camera_labels"
   ```
   Run the script using Python to generate the `front_images` directory.
   ```shell
   python data_utils/create_front_img.py
   ```

2. Set the value of `init_img_dir` in the `config/3dgs-waymo-9653.yaml` file to `front_images`.

### Set LLM API Key

Modify parameters in the `chatsim/agents/llm_utils.py` file to set your LLM API Key, such as:
```python
token = "your-github-token"
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1-mini"
```

### LLM model preparation (optional)
If you want to deploy the LLM model service locally, before starting simulation, you should start llm service.

Taking the [Qwen2.5-Coder-14B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-Coder-14B-Instruct/) model as an example, you can use the following command to download the `Qwen2.5-Coder-14B-Instruct` model weights from modelscope, which will require approximately 28GB of storage space.
```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-14B-Instruct
```

After the weights are downloaded, please modify the `run_qwen.sh` file to set the `--model` argument to the directory where the `Qwen2.5-Coder-14B-Instruct` weights are located.

Then, you can start the API server using vLLM to run the Qwen2.5-Coder-14B-Instruct model . It may cost about 40GB GPU memory.
```shell
bash run_qwen.sh
```

Modify parameters in the `chatsim/agents/llm_utils.py` file to set LLM API Key, such as:
```python
token = "EMPTY"
endpoint = "http://127.0.0.1:8300/v1"
model_name = "Qwen2.5-Coder-14B-Instruct"
```

### Start simulation
```shell
python main.py -y ${CONFIG YAML} \
               -p ${PROMPT} \
               [-s ${SIMULATION NAME}]
```

- `${CONFIG YAML}` specifies the scene information, and yamls are stored in config folder. e.g. `config/3dgs-waymo-9653.yaml`.

- `${PROMPT}`  is your input prompt, which should be wrapped in quotation marks. e.g. `add a car driving close to me`.

- `${SIMULATION NAME}` determines the name of the folder when saving results. default  `demo`.

You can try
```
python main.py -y config/3dgs-waymo-9653.yaml -p "add a car in the left front, driving close to me slowly"
```
The rendered results are saved in  `results/9653_demo_%Y_%m_%d_%H_%M_%S`. Intermediate files are saved in `results/cache/9653_demo_%Y_%m_%d_%H_%M_%S` for debug and visualization if `save_cache` are enabled in `config/waymo-9653.yaml`.

**Note** 
1. The original author's sampling strategy for insertion points has a certain degree of randomness, which may result in unsuitable points being selected. This can cause the vehicle trajectory generation to fail, leading to videos where no vehicles are inserted—indicated by `scene.added_cars_dict dict_keys([])` being empty. In such cases, simply rerun the command a few more times to resolve the issue.
2. If you encounter an error similar to `FileNotFoundError: No such file: xxxxxxx/blender_output/0/RGB_composite.png` during the generation process, please refer to this[issue](https://github.com/yifanlu0227/ChatSim/issues/40) for resolution.
3. During the generation process, Blender will launch 5 threads to perform scene rendering using the CPU. The entire rendering process takes approximately 5 minutes.