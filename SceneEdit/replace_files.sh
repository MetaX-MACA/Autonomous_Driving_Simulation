#!/bin/bash
cp ./metax_modifications/chatsim/agents/asset_select_agent.py                               ./ChatSim/chatsim/agents
cp ./metax_modifications/chatsim/agents/deletion_agent.py                                   ./ChatSim/chatsim/agents 
cp ./metax_modifications/chatsim/agents/llm_utils.py                                        ./ChatSim/chatsim/agents
cp ./metax_modifications/chatsim/agents/motion_agent.py                                     ./ChatSim/chatsim/agents
cp ./metax_modifications/chatsim/agents/project_manager.py                                  ./ChatSim/chatsim/agents
cp ./metax_modifications/chatsim/agents/view_adjust_agent.py                                ./ChatSim/chatsim/agents
cp ./metax_modifications/chatsim/foreground/Blender/utils/blender_utils/render/render.py    ./ChatSim/chatsim/foreground/Blender/utils/blender_utils/render
cp ./metax_modifications/chatsim/foreground/motion_tools/placement_and_motion.py            ./ChatSim/chatsim/foreground/motion_tools
cp ./metax_modifications/chatsim/foreground/motion_tools/placement_iterative.py             ./ChatSim/chatsim/foreground/motion_tools
cp ./metax_modifications/chatsim/foreground/motion_tools/tools.py                           ./ChatSim/chatsim/foreground/motion_tools
cp ./metax_modifications/chatsim/scene.py                                                   ./ChatSim/chatsim
cp ./metax_modifications/config/3dgs-waymo-9653.yaml                                        ./ChatSim/config
cp ./metax_modifications/data_utils/convert_to_waymo_script.py                              ./ChatSim/data_utils
cp ./metax_modifications/data_utils/create_front_img.py                                     ./ChatSim/data_utils
cp ./metax_modifications/data_utils/process_waymo.py                                        ./ChatSim/data_utils
cp ./metax_modifications/data_utils/process_waymo_script.py                                 ./ChatSim/data_utils
cp ./metax_modifications/main.py                                                            ./ChatSim
cp ./metax_modifications/requirements.txt                                                   ./ChatSim
