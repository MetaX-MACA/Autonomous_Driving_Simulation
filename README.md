# MetaX-Autonomous Driving Simulation Solution

English | [中文版](README_zh.md)

![logo](./imgs/logo.png)

## About
This solution focuses on the field of autonomous driving research, leveraging the powerful training and inference capabilities of the MetaX's C-series GPUs to build an efficient data generation and augmentation pipeline.First, 3D Gaussian Splatting (3DGS) technology is used to perform high-fidelity reconstruction of real-world scenes, rapidly restoring their 3D structures and details. Then, a Large Language Model (LLM) is introduced to semantically understand and edit the scene based on instruction-driven inputs, enabling object-level manipulation. Finally, Stable Diffusion is employed for scene style transfer, allowing the same scene to be flexibly adapted to different weather conditions, times of day, or visual styles.Through the organic integration and flexible orchestration of these three key modules, this solution enables instance-level, on-demand editing and enhancement of a single scene, such as: removing specific objects, adding new elements, enriching traffic compositions, or simulating diverse environmental and climate conditions.

This workflow significantly improves the diversity and coverage of autonomous driving datasets, especially in handling edge cases and extreme scenarios. By constructing and editing targeted scenes, it effectively addresses issues like sparse data distribution and insufficient sample volume common in traditional data collection methods.This mechanism not only provides more generalizable and robust data support for downstream tasks such as perception and decision-making, but also establishes a flexible and controllable high-quality data generation pathway for training and testing autonomous driving algorithms—ultimately advancing the overall performance of autonomous driving systems.

Demo

![demo](./imgs/demo.mp4)

## Available Solutions
### 1. Scene Reconstruction
- **[Scene Reconstruction Based on 3DGS](./SceneReconstruction/README.md)** Utilizes the capabilities of 3DGS for scene reconstruction, enabling dynamic high-fidelity modeling of urban street environments through foreground-background separation, independent modeling of rigid and non-rigid objects, and more. This lays a solid foundation for flexible scene editing.

### 2. Scene Editing
- **[Scene Editing Based on LLM](./SceneEdit/README.md)** Leverages Large Language Models (LLMs) for semantic understanding and instruction-driven editing of scene content. Object-level operations can be performed, allowing scenes to be flexibly and precisely modified based on complex natural language commands.

### 3. Scene Transition
- **[Scene Transfer Based on Stable Diffusion](./SceneTransition/README.md)** Uses Stable Diffusion for scene style transfer, enabling a single scene to be easily converted into different versions with varied weather, time of day, or visual aesthetics.

## Installation

Please refer to [install.md](./docs/install.md) for environment setup.

## Usage

Please refer to [usage.md](./docs/usage.md) for usage.

## License

This project is released under the [Apache License Version 2.0](./LICENSE). Contributions and usage are warmly welcomed.
