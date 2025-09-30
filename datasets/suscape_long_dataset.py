import os

import torch
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class SuscapeLongDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        label_dir = 'data/suscape_long_videos/third_level_labels'
        test_file = 'data/suscape_long_videos/test_list.txt'
        train_file = 'data/suscape_long_videos/train_list.txt'

        if split_type == 'train':
            self.vids = read_lines(train_file)
        else:
            self.vids = read_lines(test_file)
            
        self.video_label_tuples = []
        
        
        label_index_map = {}
        label_desc_map = {}
        descs = []


        with open("data/suscape_long_videos/label_desc_map.txt", "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(": ", 1)
                    label_index_map[key] = len(label_index_map)
                    label_desc_map[key] = value
                    descs.append(value)

        self.label_cnt = len(label_index_map)
        self.label_index_map = label_index_map
        self.label_desc_map = label_desc_map
        self.descs = descs

        for vid in self.vids:
            label_file = os.path.join(label_dir, f'{vid}.json')            
            video_info = load_json(label_file)
            
            self.video_label_tuples.append([
                video_info['data_file'],
                video_info['start_time'],
                video_info['end_time'],
                video_info['third_level_tag']
                ])

    def __getitem__(self, index):
        video_info = self.video_label_tuples[index]
        video_file, start_time, end_time, third_level_tag = video_info
            
        video_path = os.path.join(self.videos_dir, video_file + ".mp4")
        video_desc = self.label_desc_map.get(third_level_tag, "")
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type,
                                                         start_time=start_time,
                                                         end_time=end_time
                                                         )
        
        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        # Create a one-hot label tensor
        label_idx = self.label_index_map[third_level_tag]
        labels = torch.zeros(self.label_cnt, dtype=torch.float)
        labels[label_idx] = 1.0

        ret = {
            'video_id': video_path,
            'video': imgs,
            'text': video_desc,
            'labels': labels
        }

        return ret

    def __len__(self):
        return len(self.video_label_tuples)
