from torchvision import transforms
from PIL import Image


def init_transform_dict(input_res=224):
    tsfm_dict = {
        "clip_test": transforms.Compose(
            [
                transforms.Resize(input_res, interpolation=Image.BICUBIC),
                transforms.CenterCrop(input_res),
                transforms.Normalize([0.429, 0.363, 0.425], [1.396, 1.411, 1.399]),
            ]
        ),
        "clip_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0, saturation=0, hue=0),
                transforms.Normalize([0.429, 0.363, 0.425], [1.396, 1.411, 1.399]),
            ]
        ),
    }

    return tsfm_dict
