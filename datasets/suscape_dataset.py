import json
import os

import torch
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class SuscapeDataset(Dataset):
    """
    videos_dir: directory where all videos are stored
    config: AllConfig object
    split_type: 'train'/'test'
    img_transforms: Composition of transforms
    Notes: for test split, we return one video, caption pair for each caption belonging to that video
    so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type="train", img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        # TODO: update file path load method
        test_file = "data/suscape/test_list_new_filtered.json"
        train_file = "data/suscape/train_list_new_filtered.json"

        if split_type == "train":
            with open(train_file, "r") as train_file_data:
                self.video_list = json.load(train_file_data)
        else:
            with open(test_file, "r") as test_file_data:
                self.video_list = json.load(test_file_data)

    def __getitem__(self, index):
        video_info = self.video_list[index]
        video_id = video_info["video_id"]
        video_desc = video_info["desc"]
        video_path = os.path.join(self.videos_dir, video_id + ".mp4")
        imgs, idxs = VideoCapture.load_frames_from_video(
            video_path, self.config.num_frames, self.config.video_sample_type
        )

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            "video_id": video_path,
            "video": imgs,
            "text": video_desc,
        }

        return ret

    def __len__(self):
        return len(self.video_list)
