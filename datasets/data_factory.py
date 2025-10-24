from random import shuffle
from config.base_config import Config
from datasets.transform_generators import get_train_img_tfms, get_test_img_tfms
from datasets.suscape_dataset import SuscapeDataset
from datasets.nuscene_corruption_dataset import NusceneCorruptionDataset
from datasets.mixed_dataset import MixedDataset
from torch.utils.data import DataLoader


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type="train"):
        if split_type == "train":
            img_tfms = get_train_img_tfms(config.input_res)
        else:
            img_tfms = get_test_img_tfms(config.input_res)

        if config.dataset_name == "suscape":
            dataset = SuscapeDataset(config, split_type, img_tfms)
        elif config.dataset_name == "nuscene_c":
            dataset = NusceneCorruptionDataset(config, split_type, img_tfms)
        elif config.dataset_name == "mixed":
            dataset = MixedDataset(config, split_type, img_tfms)
        else:
            raise NotImplementedError(f"Dataset {config.dataset_name} is not defined")

        shuffle = split_type == "train"
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )
