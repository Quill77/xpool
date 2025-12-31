from torchvision import transforms
from PIL import Image


get_test_img_tfms = lambda input_res: transforms.Compose(
    [
        transforms.Resize(input_res, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_res),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

get_train_img_tfms = lambda input_res: transforms.Compose(
    [
        transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0, saturation=0, hue=0),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)
