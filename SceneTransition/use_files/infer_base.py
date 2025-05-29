import sys
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import argparse
import torch
from misc_utils.train_utils import unit_test_create_model
from misc_utils.image_utils import save_tensor_to_gif, save_tensor_to_images

from pl_trainer.inference.inference import InferenceIP2PVideo, InferenceIP2PVideoOpticalFlow
from dataset.single_video_dataset import SingleVideoDataset
from IPython.display import Image

def split_batch(cond, frames_in_batch=16, num_ref_frames=4):
    frames_in_following_batch = frames_in_batch - num_ref_frames
    conds = [cond[:, :frames_in_batch]]
    frame_ptr = frames_in_batch
    num_ref_frames_each_batch = []

    while frame_ptr < cond.shape[1]:
        remaining_frames = cond.shape[1] - frame_ptr
        if remaining_frames < frames_in_batch:
            frames_in_following_batch = remaining_frames
        else:
            frames_in_following_batch = frames_in_batch - num_ref_frames
        this_ref_frames = frames_in_batch - frames_in_following_batch
        conds.append(cond[:, frame_ptr:frame_ptr+frames_in_following_batch])
        frame_ptr += frames_in_following_batch
        num_ref_frames_each_batch.append(this_ref_frames)

    return conds, num_ref_frames_each_batch

class insv2v_model():
    def __init__(self, args):

        config_path = os.path.join(CURRENT_DIR, 'configs/instruct_v2v_inference.yaml')
        self.diffusion_model = unit_test_create_model(config_path)

        ckpt = torch.load(os.path.join(args.ckpt_path, 'insv2v.pth'), map_location='cpu')
        self.diffusion_model.load_state_dict(ckpt, strict=False)
        self.diffusion_model.half()
        self.diffusion_model.eval()


    def video_infer(self, VIDEO_PATH, prompt, save_path=None, IMGSIZE=(576, 128), NUM_FRAMES=50):
        EDIT_PROMPT = prompt
        VIDEO_CFG = 1.2
        TEXT_CFG = 7.5
        LONG_VID_SAMPLING_CORRECTION_STEP = 0.5

        # IMGSIZE = (576, 128)
        # NUM_FRAMES = 50
        VIDEO_SAMPLE_RATE = 24

        # sampling params
        FRAMES_IN_BATCH = 16
        NUM_REF_FRAMES = 4
        USE_MOTION_COMPENSATION = True

        if USE_MOTION_COMPENSATION:
            inf_pipe = InferenceIP2PVideoOpticalFlow(
                unet = self.diffusion_model.unet,
                num_ddim_steps=50,
                scheduler='ddpm'
            )
            
        else:
            inf_pipe = InferenceIP2PVideo(
                unet = self.diffusion_model.unet,
                num_ddim_steps=50,
                scheduler='ddpm'
            )

        dataset = SingleVideoDataset(
            video_file=VIDEO_PATH,
            video_description='',
            sampling_fps=VIDEO_SAMPLE_RATE,
            num_frames=NUM_FRAMES,
            output_size=IMGSIZE
        )

        batch = dataset[0] # start from 20th frame
        batch = {k: v.cuda()[None] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        cond = [self.diffusion_model.encode_image_to_latent(frames) / 0.18215 for frames in batch['frames'].chunk(16, dim=1)] # when encoding, chunk the frames to avoid oom in vae, you can reduce the 16 if you have a smaller gpu
        
        cond = torch.cat(cond, dim=1).to(torch.float16)
        text_cond = self.diffusion_model.encode_text([EDIT_PROMPT]).to(torch.float16)
        text_uncond = self.diffusion_model.encode_text(['']).to(torch.float16)
        conds, num_ref_frames_each_batch = split_batch(cond, frames_in_batch=FRAMES_IN_BATCH, num_ref_frames=NUM_REF_FRAMES)
        splitted_frames, _ = split_batch(batch['frames'], frames_in_batch=FRAMES_IN_BATCH, num_ref_frames=NUM_REF_FRAMES)

        # First video clip
        cond1 = conds[0]
        latent_pred_list = []
        init_latent = torch.randn_like(cond1).to(torch.float16)
        
        latent_pred = inf_pipe(
            latent = init_latent,
            text_cond = text_cond,
            text_uncond = text_uncond,
            img_cond = cond1,
            text_cfg = TEXT_CFG,
            img_cfg = VIDEO_CFG,
        )['latent']
        latent_pred_list.append(latent_pred)

        # Subsequent video clips
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for prev_cond, cond_, prev_frame, curr_frame, num_ref_frames_ in zip(
                conds[:-1], conds[1:], splitted_frames[:-1], splitted_frames[1:], num_ref_frames_each_batch
            ):
                init_latent = torch.cat([init_latent[:, -num_ref_frames_:], torch.randn_like(cond_)], dim=1)
                cond_ = torch.cat([prev_cond[:, -num_ref_frames_:], cond_], dim=1)
                if USE_MOTION_COMPENSATION:
                    ref_images = prev_frame[:, -num_ref_frames_:]
                    query_images = curr_frame
                    additional_kwargs = {
                        'ref_images': ref_images,
                        'query_images': query_images,
                    }
                else:
                    additional_kwargs = {}
                latent_pred = inf_pipe.second_clip_forward(
                    latent = init_latent, 
                    text_cond = text_cond,
                    text_uncond = text_uncond,
                    img_cond = cond_,
                    latent_ref = latent_pred[:, -num_ref_frames_:],
                    noise_correct_step = LONG_VID_SAMPLING_CORRECTION_STEP,
                    text_cfg = TEXT_CFG,
                    img_cfg = VIDEO_CFG,
                    **additional_kwargs,
                )['latent']
                latent_pred_list.append(latent_pred[:, num_ref_frames_:])

            # Save GIF
            latent_pred = torch.cat(latent_pred_list, dim=1)
            image_pred = self.diffusion_model.decode_latent_to_image(latent_pred).clip(-1, 1)

            original_images = batch['frames'].cpu()
            transferred_images = image_pred.float().cpu()
            concat_images = torch.cat([original_images, transferred_images], dim=4)
            if save_path is not None:
                self.save_2_gif(concat_images, transferred_images, save_path)

            return transferred_images

    def save_2_gif(self, concat_images, transferred_images, save_path):

        save_tensor_to_gif(concat_images, os.path.join(save_path, 'video_edit.gif'), fps=5)
        save_tensor_to_images(transferred_images, os.path.join(save_path, 'video_edit_images'))

        # visualize the gif
        Image(filename=os.path.join(save_path, 'video_edit.gif'))


def build_video(image_folder, video_name):    
    import cv2
    import os
    fps = 16

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  

    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        image = cv2.imread(img_path)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()
    print(f"video saved: {video_name}")



def main(args):
    read_fold = args.video_fold
    video_list = os.listdir(read_fold)
    print(video_list)
    
    prompt = args.prompt
    IMGSIZE = args.image_size
    
    NUM_FRAMES = args.num_frames
    save_fold = args.save_fold
    if not os.path.isdir(save_fold):
        os.makedirs(save_fold, exist_ok=True)
    
    for file in video_list:
        model = insv2v_model(args)
        name = file.split('.')[0]
        video = os.path.join(read_fold, file)
        save_path = os.path.join(save_fold, name)
        out = model.video_infer(video, prompt, save_path, IMGSIZE)
        build_video(os.path.join(save_path, 'video_edit_images'), os.path.join(save_path, 'night.mp4'))
        print('{} infer done!'.format(name))

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Your program description')

    # Add arguments
    parser.add_argument('--video_fold', type=str, default='/mnt/data/fdeng/code/videos_fold', help='the fold of videos need to edit')
    parser.add_argument('--prompt', type=str, default='make sunny sky to dark night.', help='edit requirements')
    parser.add_argument('--save_fold', type=str, default='./results', help='save address for finished edit')
    parser.add_argument('--image-size', type=int, nargs=2, default=[480, 320], help='Image size')
    parser.add_argument('--num_frames', type=int, default=60, help='Prompt source')
    parser.add_argument('--ckpt-path', type=str, default='/mnt/data/fdeng/work_temp/instruct-video-to-video/weights', help='Path to checkpoint')

    # Parse arguments
    args = parser.parse_args()

    main(args)

