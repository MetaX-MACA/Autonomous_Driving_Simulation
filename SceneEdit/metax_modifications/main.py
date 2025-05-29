# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import open3d as o3d # 额外添加的import语句，用于解决C500环境下scene文件中 import失败问题
import argparse
import openai
from termcolor import colored
import imageio.v2 as imageio
import pprint
from chatsim.agents.asset_select_agent import AssetSelectAgent
from chatsim.agents.background_rendering_agent import BackgroundRenderingAgent
from chatsim.agents.background_rendering_3dgs_agent import BackgroundRendering3DGSAgent
from chatsim.agents.deletion_agent import DeletionAgent
from chatsim.agents.foreground_rendering_agent import ForegroundRenderingAgent
from chatsim.agents.motion_agent import MotionAgent
from chatsim.agents.project_manager import ProjectManager
from chatsim.agents.view_adjust_agent import ViewAdjustAgent
from chatsim.agents.utils import generate_video, read_yaml
from chatsim.scene import Scene
import os
from glob import glob 
from tqdm import tqdm
import cv2

def get_parser():
    parser = argparse.ArgumentParser(description="ChatSim argrument parser.")
    parser.add_argument(
        "--config_yaml", "-y", type=str,
        default="config/waymo-1287.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        default="add a straight driving car in the scene",
        help="language prompt to ChatSim.",
    )
    parser.add_argument(
        "--simulation_name", "-s", type=str,
        default="demo",
        help="simulation experiment name.",
    )

    args = parser.parse_args()
    return args


class ChatSim:
    def __init__(self, config):
        self.config = config

        self.scene = Scene(config["scene"])  # agents share and maintain the same scene

        agents_config = config['agents']
        self.project_manager = ProjectManager(agents_config["project_manager"])
        self.asset_select_agent = AssetSelectAgent(agents_config["asset_select_agent"])
        self.deletion_agent = DeletionAgent(agents_config["deletion_agent"])
        self.foreground_rendering_agent = ForegroundRenderingAgent(agents_config["foreground_rendering_agent"])
        self.motion_agent = MotionAgent(agents_config["motion_agent"])
        self.view_adjust_agent = ViewAdjustAgent(agents_config["view_adjust_agent"])
        # we can choose between nerf and 3dgs for background rendering
        if agents_config['background_rendering_agent'].get("scene_representation", 'nerf') == 'nerf':
            self.background_rendering_agent = BackgroundRenderingAgent(agents_config["background_rendering_agent"])
        else:
            self.background_rendering_agent = BackgroundRendering3DGSAgent(agents_config["background_rendering_agent"])

        self.tech_agents = {
            "asset_select_agent": self.asset_select_agent,
            "background_rendering_agent": self.background_rendering_agent,
            "deletion_agent": self.deletion_agent,
            "foreground_rendering_agent": self.foreground_rendering_agent,
            "motion_agent": self.motion_agent,
            "view_adjust_agent": self.view_adjust_agent,
        }

        self.current_prompt = (
            "An empty prompt"  # initialization place holder for debugging
        )

    def setup_init_frame(self):
        """Setup initial frame for ChatSim's reasoning and rendering.
        """
        if not os.path.exists(self.scene.init_img_path):
            print(f"{colored('[Note]', color='red', attrs=['bold'])} ",
                  f"{colored('can not find init image, rendering it for the first time')}\n")
            # it will update scene.current_images
            self.background_rendering_agent.func_render_background(self.scene)
            # save the initial image
            imageio.imwrite(self.scene.init_img_path, self.scene.current_images[0])
        else:
            self.scene.current_images = [imageio.imread(self.scene.init_img_path)] * self.scene.frames

    def setup_init_frame_other_render(self):
        img_path_list = sorted(glob(self.scene.init_img_dir + '/*.png'))
        ignore_num = 0 # 忽略最初的一些帧，默认为0
        last_index = self.scene.frames + ignore_num
        img_path_list = img_path_list[ignore_num:last_index]
        self.scene.current_images = []
        print("Load blended images")
        for item in tqdm(img_path_list):
            img = cv2.imread(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.scene.current_images.append(img)


    def execute_llms(self, prompt):
        """Entry of ChatSim's reasoning.
        We perform multi-LLM reasoning for the user's prompt

        Input:
            prompt : str
                language prompt to ChatSim.
        """
        self.scene.setup_cars()
        self.current_prompt = prompt

        # execute agent's LLM part
        tasks = self.project_manager.decompose_prompt(self.scene, prompt)

        for task in tasks.values():
            print(
                f"{colored('[Performing Single Prompt]', on_color='on_blue', attrs=['bold'])} {colored(task, attrs=['bold'])}\n"
            )
            self.project_manager.dispatch_task(self.scene, task, self.tech_agents)

        print(colored("scene.added_cars_dict", color="red", attrs=["bold"]), end=' ')
        pprint.pprint(self.scene.added_cars_dict.keys())
        print(colored("scene.removed_cars", color="red", attrs=["bold"]), end=' ')
        pprint.pprint(self.scene.removed_cars)

    def execute_funcs(self):
        """Entry of ChatSim's rendering functions
        We perform agent's functions following the self.scene's configuration.
        self.scene's configuration are updated in self.execute_llms()
        """
        # use scene.current_extrinsics, render (novel) view images
        # self.background_rendering_agent.func_render_background(self.scene) # 使用其他模型渲染的图像，不需要进行此步骤

        # Inpaint. 
        self.deletion_agent.func_inpaint_scene(self.scene)

        # Retrieve blender file from asset bank
        self.asset_select_agent.func_retrieve_blender_file(self.scene)

        # Blender add car. If no addition, just return
        self.foreground_rendering_agent.func_blender_add_cars(self.scene)

        # Generate Video
        generate_video(self.scene, self.current_prompt)


if __name__ == "__main__":
    args = get_parser()

    config = read_yaml(args.config_yaml)
    config['scene']["simulation_name"] = args.simulation_name

    chatsim = ChatSim(config)
    # chatsim.setup_init_frame()
    chatsim.setup_init_frame_other_render()
    chatsim.execute_llms(args.prompt)
    chatsim.execute_funcs()
