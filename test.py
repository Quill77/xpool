import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules import metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed < 0:
        print("Random seed not set.")
        return
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metrics(metric_name):
    if metric_name == "t2v":
        return metrics.t2v_metrics
    elif metric_name == "v2t":
        return metrics.v2t_metrics
    else:
        raise NotImplementedError(f"Metric {metric_name} not defined.")


def get_tokenizer(use_huggingface):
    if not use_huggingface:
        from modules.tokenization_clip import SimpleTokenizer

        return SimpleTokenizer()

    else:
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(
            "/lab/haoq_lab/12532563/xpool/checkpoints/clip-vit-base-patch32",
            TOKENIZERS_PARALLELISM=False,
        )


def main():
    config = AllConfig()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(config.seed)

    model = ModelFactory.get_model(config)
    loss = LossFactory.get_loss(config)
    metrics = get_metrics(config.metric)

    test_data_loader = DataFactory.get_data_loader(config, split_type="test")

    writer = (
        None if not config.no_tensorboard else SummaryWriter(log_dir=config.tb_log_dir)
    )
    tokenizer = get_tokenizer(use_huggingface=config.huggingface)

    trainer = Trainer(
        model,
        loss,
        metrics,
        None,
        config=config,
        train_data_loader=None,
        valid_data_loader=test_data_loader,
        scheduler=None,
        writer=writer,
        tokenizer=tokenizer,
    )

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")
    trainer.validate()


if __name__ == "__main__":
    main()
