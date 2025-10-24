from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import (
    sim_matrix_training,
    sim_matrix_inference,
    generate_embeds_per_video_id,
)
from tqdm import tqdm


def min_max_normalize(sims):
    """
    Min-max normalize the similarity scores to [0, 1]
    """
    sims_min = sims.min()
    sims_max = sims.max()
    normalized_sims = (sims - sims_min) / (sims_max - sims_min + 1e-8)
    return normalized_sims


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        config: Config,
        train_data_loader,
        valid_data_loader,
        tokenizer,
        scheduler=None,
        writer=None,
    ):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def _process_text(self, text):
        if self.tokenizer is not None:
            text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        else:
            text = {key: val.to(self.device) for key, val in text.items()}
        return text

    def _eval(self, epoch, batch_idx, num_steps):
        val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)

        if val_res["R1-window"] > self.best_window:
            self.best_window = val_res["R1-window"]
            self._save_checkpoint(epoch, save_best=True)

        if val_res["R1"] > self.best:
            self.best = val_res["R1"]

        print(" Current Best Window Average R@1 is {}".format(self.best_window))
        print(" Current Best R@1 is {}\n\n".format(self.best))

    def _get_labels(self, label_strs):
        """
        Get labels from data batch
        """
        batch_size = len(label_strs)
        label_sets = [set(label_str.split("-")) for label_str in label_strs]
        labels = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if label_sets[i] >= label_sets[j]:
                    labels[i, j] = 1.0
        return labels

    def get_loss(self, sims, label_strs):
        if self.config.loss == "circle":
            labels = self._get_labels(label_strs)
            sims = min_max_normalize(sims)
            loss = self.loss(sims, labels)
        else:
            loss = self.loss(sims, self.model.clip.logit_scale)
        return loss

    def _log(self, epoch, batch_idx, num_steps, loss_val, eval_steps):
        if self.writer is not None:
            self.writer.add_scalar("train/loss_train", loss_val, self.global_step)

        if batch_idx % self.log_step == 0:
            print("Train Epoch: {} dl: {}/{} Loss: {:.6f}".format(epoch, batch_idx, num_steps - 1, loss_val))

        if batch_idx in eval_steps:
            self._eval(epoch, batch_idx, num_steps)

    def step(self):
        """
        A single optimization step
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            # 1. clear gradients
            self.optimizer.zero_grad()

            # 2. forward pass
            data["text"] = self._process_text(data["text"])
            data["video"] = data["video"].to(self.device)

            text_embeds, video_embeds, video_embeds_pooled = self.model(data)
            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)

            # 3. compute loss
            loss = self.get_loss(sims, data["label"])

            # 4. backward pass
            loss.backward()
            loss_val = loss.detach().item()
            total_loss += loss_val

            # 5. optimizer step
            self.step()
            self._log(epoch, batch_idx, num_steps, loss_val, eval_steps)

        res = {"loss_train": total_loss / num_steps}

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        video_embed_arr = []
        video_id_arr = []
        data_label_arr = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                # 1. forward pass
                data["text"] = self._process_text(data["text"])
                data["video"] = data["video"].to(self.device)

                text_embeds, video_embed, video_embeds_pooled = self.model(data)
                sims_batch = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)

                # 2. compute loss
                loss = self.get_loss(sims_batch, data["label"])
                total_val_loss += loss.item()

                # 3. store embeddings and ids
                video_id_arr.extend(data["video_id"])
                data_label_arr.extend(data["label"])

                text_embed_arr.append(text_embeds)
                video_embed_arr.append(video_embed)

            text_embeds = torch.cat(text_embed_arr).cpu()
            video_embeds = torch.cat(video_embed_arr).cpu()
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            video_embeds_pooled = self.model.pool_frames(text_embeds, video_embeds)
            self.model.pool_frames.to(self.device)

            # text_embeds_per_video_id, video_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, video_embeds_pooled, video_id_arr, self.pooling_type)

            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)

            sims = sims.unsqueeze(1)  # num_vids x 1 x num_vids
            metrics = self.metrics
            res = metrics(sims)

            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            # New evaluation metric
            new_metrics = self._compute_new_metrics(sims, data_label_arr)
            res.update(new_metrics)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            print(
                f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                f"R@1: {res['R1']:.4f} (window: {res['R1-window']:.4f})\n",
                f"R@3: {res['R3']:.4f} (window: {res['R3-window']:.4f})\n",
                f"R@5: {res['R5']:.4f} (window: {res['R5-window']:.4f})\n",
                f"R@10: {res['R10']:.4f} (window: {res['R10-window']:.4f})\n",
                f"MedR: {res['MedR']:.4f} (window: {res['MedR-window']:.4f})\n",
                f"MeanR: {res['MeanR']:.4f} (window: {res['MeanR-window']:.4f})\n",
                f"New R@1: {res['new_R1']:.4f}\n",
                f"New R@3: {res['new_R3']:.4f}\n",
                f"New R@5: {res['new_R5']:.4f}\n",
                f"Loss: {total_val_loss}\n",
            )

            res["loss_val"] = total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f"val/{m}", res[m], self.global_step)

            self.model.train()
            return res

    def _compute_new_metrics(self, sims, data_label_arr):
        """
        Compute new evaluation metrics based on the given similarity matrix and labels
        """
        num_vids, max_text_per_vid, _ = sims.shape
        new_metrics = {"new_R1": 0, "new_R3": 0, "new_R5": 0}
        total_queries = 0

        for i in range(num_vids):
            for j in range(max_text_per_vid):
                # Get the similarity scores for the current text
                sim_scores = sims[i, j].cpu().numpy()
                # Get the top 5 retrieved video indices
                top_indices = np.argsort(-sim_scores)[:5]

                # Convert the label to a set
                query_label_set = set(data_label_arr[i * max_text_per_vid + j].split("-"))

                # Check if the retrieved videos contain the query label set
                for k, idx in enumerate(top_indices):
                    retrieved_label_set = set(data_label_arr[idx].split("-"))
                    if query_label_set.issubset(retrieved_label_set):
                        if k == 0:
                            new_metrics["new_R1"] += 1
                        if k < 3:
                            new_metrics["new_R3"] += 1
                        if k < 5:
                            new_metrics["new_R5"] += 1
                        break

                total_queries += 1

        # Normalize the metrics
        # TODO: modify this compute process
        new_metrics["new_R1"] *= 100 / total_queries
        new_metrics["new_R3"] *= 100 / total_queries
        new_metrics["new_R5"] *= 100 / total_queries

        return new_metrics
