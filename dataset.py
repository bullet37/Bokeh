import glob
import os
from typing import Callable, Optional
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
class BokehDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None,
                 train=False, test=False, validation=False, samples_train=200):
        self._root_folder = root_folder
        self._transform = transform
        self._train = train
        self._test = test
        self._validation = validation

        if self._train:
            self._source_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.src.jpg")))[:samples_train]
            self._target_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.tgt.jpg")))[:samples_train]
            self._alpha_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.alpha.png")))[:samples_train]
        elif self._validation:
            self._source_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.src.jpg")))[samples_train: ]
            self._target_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.tgt.jpg")))[samples_train: ]
            self._alpha_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.alpha.png")))[samples_train: ]
        elif self._test:
            self._source_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.src.jpg")))[:]
            self._target_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.tgt.jpg")))[:]
            self._alpha_paths = sorted(glob.glob(os.path.join(self._root_folder, "*.alpha.png")))[:]
        else:
            raise ValueError("Must specify train, validation or test mode.")

        self._meta_data = self._read_meta_data(os.path.join(root_folder, "meta.txt"))

        file_counts = [
            len(self._source_paths),
            len(self._target_paths),
            len(self._alpha_paths),
            len(self._meta_data),
        ]
        # if not file_counts[0] or len(set(file_counts)) != 1:
        #     raise ValueError(
        #         f"Empty or non-matching number of files in root dir: {file_counts}. "
        #         "Expected an equal number of source, target, source-alpha and target-alpha files. "
        #         "Also expecting matching meta file entries."
        #     )

    def __len__(self):
        return len(self._source_paths)

    def _read_meta_data(self, meta_file_path: str):
        if not os.path.isfile(meta_file_path):
            raise ValueError(f"Meta file missing under {meta_file_path}.")

        meta = {}
        with open(meta_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
            meta[id] = (src_lens, tgt_lens, disparity)
        return meta

    def __getitem__(self, index):
        to_tensor = transforms.ToTensor()
        source = to_tensor(Image.open(self._source_paths[index]))
        input_resolution = source.shape

        target = to_tensor(Image.open(self._target_paths[index]))
        alpha = to_tensor(Image.open(self._alpha_paths[index]))

        filename = os.path.basename(self._source_paths[index])
        id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = self._meta_data[id]

      
        src_lens_type = torch.tensor(0,dtype=torch.float32) if src_lens.split("50mmf")[0] == 'Canon' else torch.tensor(1,dtype=torch.float32)    # Sony: 1, Canon: 0
        tgt_lens_type = torch.tensor(0,dtype=torch.float32) if tgt_lens.split("50mmf")[0] == 'Canon' else torch.tensor(1,dtype=torch.float32)  # Sony: 1, Canon: 0
        src_F = torch.tensor(float(src_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        tgt_F = torch.tensor(float(tgt_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        disparity = torch.tensor(float(disparity), dtype=torch.float32) / 100


        if self._transform:
            source = self._transform(source)
            target = self._transform(target)
            alpha = self._transform(alpha)

        return {
            "source": source,
            "target": target,
            "alpha": alpha,
            "src_lens": src_lens,
            "tgt_lens": tgt_lens,
            "disparity": disparity,
            "src_lens_type": src_lens_type,
            "tgt_lens_type": tgt_lens_type,
            "src_F": src_F,
            "tgt_F": tgt_F,
            "image_id": [id],
            "resolution": [input_resolution]
        }



class CustomTransformer:
    def __init__(self, apply_flip=True, apply_color_jitter=False, apply_random_crop=False):
        self.apply_flip = apply_flip
        self.apply_color_jitter = apply_color_jitter
        self.apply_random_crop = apply_random_crop


        self.flip = transforms.RandomHorizontalFlip(p=1.0)  # 确保每次都翻转
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.crop = transforms.RandomCrop(size=(200, 200))  # 假设所有图像至少是200x200大小

    def __call__(self, data):
        if self.apply_flip:
            data['source'] = self.flip(data['source'])
            data['target'] = self.flip(data['target'])
            data['alpha'] = self.flip(data['alpha'])

        if self.apply_color_jitter:
            data['source'] = self.color_jitter(data['source'])
            data['target'] = self.color_jitter(data['target'])
            # 通常 alpha 通道不应用颜色抖动

        if self.apply_random_crop:
            # 获取随机裁剪的参数
            i, j, h, w = self.crop.get_params(data['source'], self.crop.size)
            data['source'] = transforms.functional.crop(data['source'], i, j, h, w)
            data['target'] = transforms.functional.crop(data['target'], i, j, h, w)
            data['alpha'] = transforms.functional.crop(data['alpha'], i, j, h, w)

        return data
