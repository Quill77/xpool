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
from modules.optimization import AdamW, get_cosine_schedule_with_warmup


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


def get_optimizer(model, config):
    model_params = list(model.named_parameters())
    clip_params = [p for n, p in model_params if "clip." in n]
    noclip_params = [p for n, p in model_params if "clip." not in n]

    optimizer_grouped_params = [
        {"params": clip_params, "lr": config.clip_lr},
        {"params": noclip_params, "lr": config.noclip_lr},
    ]
    return AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)


def get_scheduler(config, train_data_loader, optimizer):
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


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
    optimizer = get_optimizer(model, config)

    train_data_loader = DataFactory.get_data_loader(config, split_type="train")
    valid_data_loader = DataFactory.get_data_loader(config, split_type="test")

    scheduler = get_scheduler(config, train_data_loader, optimizer)

    writer = None if config.no_tensorboard else SummaryWriter(log_dir=config.tb_log_dir)
    tokenizer = get_tokenizer(use_huggingface=config.huggingface)

    trainer = Trainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        scheduler=scheduler,
        writer=writer,
        tokenizer=tokenizer,
    )

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("msrvtt_9k_model_best.pth")
    trainer.train()


if __name__ == "__main__":
    main()
