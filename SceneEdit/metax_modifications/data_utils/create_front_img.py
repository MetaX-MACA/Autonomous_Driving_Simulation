import os
from glob import glob
from tqdm import tqdm
from shutil import copyfile

waymo_data_dir = ""


def move_image():
    src_dir = os.path.join(waymo_data_dir, 'images_all')
    save_dir = os.path.join(waymo_data_dir, 'front_images')
    os.makedirs(save_dir, exist_ok=True)

    img_path_list = sorted(glob(src_dir + '/*.png'))
    for item in tqdm(img_path_list):
        if '_FRONT.png' in item:
            save_path = item.replace(src_dir, save_dir)
            copyfile(item, save_path)

if __name__ == '__main__':
    move_image()