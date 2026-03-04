import numpy as np
import torch

import time
import os, hashlib

from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, compute_labeled_metrics

def _save(y_true, y_pred, save_path="pred_labels.txt"):
    with open(save_path, "w") as f:
        for true_label, pred_label in zip(y_true, y_pred):
            f.write(f"{true_label}, {pred_label}\n")

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
        self.best = -1.0

    def _process_text(self, text):
        if self.tokenizer is not None:
            text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        else:
            text = {key: val.to(self.device) for key, val in text.items()}
        return text

    def _get_containments(self, label_strs):
        """
        Get labels from data batch
        """
        batch_size = len(label_strs)
        label_sets = [set(label_str.split("-")) for label_str in label_strs]
        containments = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if label_sets[i] >= label_sets[j]:
                    containments[i, j] = 1.0
        return containments

    def _get_loss(self, sims, label_strs=None):
        n = sims.size(0)
        if self.config.loss in ("circle", "mix"):
            containments = torch.eye(n, dtype=torch.bool, device=self.device) if label_strs is None else self._get_containments(label_strs)

        if self.config.loss == "circle":
            loss = self.loss(sims, containments)
        elif self.config.loss == "clip":
            loss = self.loss(sims, self.model.clip.logit_scale)
        elif self.config.loss == "mix":
            loss = self.loss(sims, containments, self.model.clip.logit_scale)
        return loss

    def _log(self, epoch, batch_idx, num_steps, loss_val):
        if self.writer is not None:
            self.writer.add_scalar("train/loss_train", loss_val, self.global_step)

        if batch_idx % self.log_step == 0:
            print("Train Epoch: {} dl: {}/{} Loss: {:.6f}".format(epoch, batch_idx, num_steps - 1, loss_val))

    def _step(self):
        """
        A single optimization step
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

    def _train(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)

        for batch_idx, data in enumerate(self.train_data_loader):
            # 1. clear gradients
            self.optimizer.zero_grad()

            # 2. forward pass
            data["text"] = self._process_text(data["text"])
            data["video"] = data["video"].to(self.device)

            text_embeds, video_embeds, video_embeds_pooled = self.model(data)
            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)

            # 3. compute loss
            loss = self._get_loss(sims, data["label"] if "label" in data else None)

            # 4. backward pass
            loss.backward()
            loss_val = loss.detach().item()
            total_loss += loss_val

            # 5. optimizer step
            self._step()
            self._log(epoch, batch_idx, num_steps, loss_val)

        if epoch % self.config.eval_every == 0:
            val_res = self._valid()

            if val_res["R1"] > self.best:
                self.best = val_res["R1"]

            print(" Current Best R@1 is {}\n".format(self.best))

        if epoch % self.config.save_every == 0:
            self._save_checkpoint(epoch, save_best=False)

            if epoch % self.config.save_every == 0:
                    self._save_checkpoint(epoch, save_best=False)
        res = {"loss_train": total_loss / num_steps}
        return res

    def _valid(self):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        print(f"---------------Validation---------------\n")
        self.model.eval()
        text_embed_arr = []
        video_embed_arr = []
        video_id_arr = []
        data_label_arr = []

        with torch.no_grad():
            for _, data in enumerate(self.valid_data_loader):
                # 1. forward pass
                data["text"] = self._process_text(data["text"])
                data["video"] = data["video"].to(self.device)

                text_embeds, video_embed, video_embeds_pooled = self.model(data)

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

            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            sims = sims.unsqueeze(1)

            res = self.metrics(sims)

            sims = sims.squeeze(1)
            data_label_arr = [sorted(label.split("-"))[-1] if label.startswith("1") or label.startswith("2") else label for label in data_label_arr]
            y_true = np.array(data_label_arr)
            y_pred = np.array([y_true[i] for i in sims.argmax(axis=1)])
            _save(y_true, y_pred)

            # New evaluation metric
            new_metrics = compute_labeled_metrics(sims, data_label_arr)
            res.update(new_metrics)

            print(
                f"R@1: {res['R1']:.4f}\n",
                f"R@3: {res['R3']:.4f}\n",
                f"R@5: {res['R5']:.4f}\n",
                f"LR@1: {res['new_R1']:.4f}\n",
                f"LR@3: {res['new_R3']:.4f}\n",
                f"LR@5: {res['new_R5']:.4f}\n",
                f"LP@1: {res['new_P1']:.4f}\n",
                f"LP@3: {res['new_P3']:.4f}\n",
                f"LP@5: {res['new_P5']:.4f}\n",
            )

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f"val/{m}", res[m], self.global_step)

            self.model.train()
            return res

    def _valid_v2v(self):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        print(f"---------------Validation---------------\n")
        self.model.eval()
        video_embed_arr = []
        data_label_arr = []

        with torch.no_grad():
            # for _, data in tqdm(enumerate(self.valid_data_loader)):
            for _, data in enumerate(self.valid_data_loader):
                # 1. forward pass
                # data["text"] = data["video"].to(self.device)
                data["video"] = data["video"].to(self.device)

                batch_size = data["video"].shape[0]

                video_data = data["video"]

                video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

                video_embeds = self.model.clip.get_image_features(video_data)

                video_embeds = video_embeds.reshape(batch_size, self.config.num_frames, -1)

                # query_embeds = self.pool_frames(video_embeds, video_embeds)
                # video_embeds_pooled = self.pool_frames(query_embeds, video_embeds)

                # sims = sim_matrix_training(query_embeds, video_embeds_pooled, self.pooling_type)

                # 2. compute loss
                # loss = self.get_loss(sims, data["label"])
                # total_val_loss += loss.item()

                # 3. store embeddings and ids
                # video_id_arr.extend(data["video_id"])
                data_label_arr.extend(data["label"])

                video_embed_arr.append(video_embeds)

            video_embeds = torch.cat(video_embed_arr).cpu()
            # for i in range(1, video_embeds.shape[1] + 1):
            i = 8
            query_embeds = video_embeds[:, :, :]
            # query_embeds = video_embeds[:, torch.randperm(video_embeds.size(1))[:video_embeds.size(1)//i], :]
            # query_embeds = query_embeds.mean(dim=1)
            # print(f"[Info] Mean query embeds shape: {query_embeds.shape}", flush=True)
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            query_embeds_mean = query_embeds.mean(dim=1).cpu()
            print(f"[Info] Query embeds mean shape: {query_embeds_mean.shape}, query_embeds shape: {query_embeds.shape}", flush=True)
            query_embeds_pooled = self.model.pool_frames(query_embeds_mean, query_embeds)
            query_embeds_pooled = query_embeds_pooled[torch.arange(query_embeds_pooled.shape[0]), torch.arange(query_embeds_pooled.shape[0]), :]
            print(f"[Info] Pooled query embeds shape: {query_embeds_pooled.shape}, Video embeds shape: {video_embeds.shape}", flush=True)
            video_embeds_pooled = self.model.pool_frames(query_embeds_pooled, video_embeds)
            self.model.pool_frames.to(self.device)

            # text_embeds_per_video_id, video_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, video_embeds_pooled, video_id_arr, self.pooling_type)

            sims = sim_matrix_training(query_embeds_pooled, video_embeds_pooled, self.pooling_type)

            sims.fill_diagonal_(-1)
            sims = sims.unsqueeze(1)  # num_vids x 1 x num_vids
            
            res = self.metrics(sims)

            # New evaluation metric
            sims = sims.squeeze(1)
            new_metrics = compute_labeled_metrics(sims, data_label_arr)
            res.update(new_metrics)

            print(
                f"R@1: {res['R1']:.4f}\n",
                f"R@3: {res['R3']:.4f}\n",
                f"R@5: {res['R5']:.4f}\n",
                f"LR@1: {res['new_R1']:.4f}\n",
                f"LR@3: {res['new_R3']:.4f}\n",
                f"LR@5: {res['new_R5']:.4f}\n",
                f"LP@1: {res['new_P1']:.4f}\n",
                f"LP@3: {res['new_P3']:.4f}\n",
                f"LP@5: {res['new_P5']:.4f}\n",
            )

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f"val/{m}", res[m], self.global_step)

            self.model.train()
            return res

    def _get_cache_path(self):
        """根据当前配置算一个唯一的缓存文件名"""
        # 把所有会影响输出 shape 的参数拼成字符串
        fingerprint = f"{self.config.input_res}_{self.config.num_frames}_{len(self.valid_data_loader.dataset)}"  # 如果你换了 CLIP 模型也要刷新缓存
        md5 = hashlib.md5(fingerprint.encode()).hexdigest()
        cache_dir = "./embed_cache"  # 想放哪里自己改
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"video_embeds_{md5}.pt")

    def _extract_video_embeds(self):
        """带缓存的提取逻辑：有缓存直接读，没有就照常算并写缓存"""
        cache_file = self._get_cache_path()
        if os.path.exists(cache_file):
            print(f"[Info] loading video embeds...")
            ckpt = torch.load(cache_file, map_location=self.device)
            return ckpt["video_embeds"], ckpt["video_id_arr"]

        print("[Info] cache failed. Start extracting video embeds...", flush=True)
        video_embed_arr, video_id_arr = [], []
        with torch.no_grad():
            for _, data in enumerate(self.valid_data_loader):
                data["video"] = data["video"].to(self.device)
                batch_size = data["video"].shape[0]
                video_data = data["video"].reshape(-1, 3, self.config.input_res, self.config.input_res)
                video_embeds = self.model.clip.get_image_features(video_data)
                video_embeds = video_embeds.reshape(batch_size, self.config.num_frames, -1)
                video_embed_arr.append(video_embeds)
                video_id_arr.extend(data["video_id"])
                print(f"[Info] Processed {_ + 1} / {len(self.valid_data_loader)} batches.", flush=True)

        video_embeds = torch.cat(video_embed_arr, dim=0)  # [N,T,d]
        torch.save({"video_embeds": video_embeds, "video_id_arr": video_id_arr}, cache_file)
        print(f"[Info] cache wrote in {cache_file}")
        return video_embeds, video_id_arr

    def v2v_retrieval(self):
        """
        Video-to-video retrieval for a single text query
        """
        print(f"---------------V2V Retrieval---------------\n")
        self.model.eval()
        video_id_arr = []

        print("[Info] Start processing videos...", flush=True)

        with torch.no_grad():
            video_embeds, video_id_arr = self._extract_video_embeds()
            self.start_v2v_retrieval_loop(video_embeds, video_id_arr)

    def start_v2v_retrieval_loop(self, video_embeds, video_id_arr):
        input_file = "v2v_input.txt"
        output_file = "v2v_output.txt"
        json_file = "data/mixed/combined_desc.json"
        import json

        with open(json_file, "r") as f:
            label_data = json.load(f)
        vid_to_label = {item["video_id"]: item["label"] for item in label_data}
        last_query = "a"
        print("[Info] Start retrieval. Waiting for input video queries...", flush=True)
        while True:
            time.sleep(1)
            with open(input_file, "r") as f:
                vid = f.read().strip()
            if vid == "":
                continue
            if vid == last_query:
                continue

            print(f"[Info] Start processing query...", flush=True)
            last_query = vid

            # Find the index of the query video
            try:
                query_index = video_id_arr.index(vid)
            except ValueError:
                print(f"[Warning] Video ID {vid} not found in the dataset.", flush=True)
                continue

            query_embeds = video_embeds[query_index : query_index + 1, ::2, :].to(self.device)
            query_embeds = query_embeds.mean(dim=1)

            # Pool frames for inference once we have all texts and videos
            video_embeds_pooled = self.model.pool_frames(query_embeds, video_embeds.to(self.device))
            sims = sim_matrix_training(query_embeds, video_embeds_pooled, self.pooling_type)
            sims = sims.squeeze(0)

            with open(output_file, "w") as f:
                f.write("Top 5 retrieved video IDs and sims:\n")
                for idx in sims.topk(5).indices:
                    rid = video_id_arr[idx]
                    sim = sims[idx].item()
                    label = vid_to_label.get(rid, ["N/A"])
                    label.sort()
                    f.write(f"video id: {rid}, sims: {sim:.4f}, label: {label}\n")

    def _retrieval(self):
        """
        Retrieval videos for a single text query
        """
        print(f"---------------Retrieval---------------\n")
        self.model.eval()
        video_id_arr = []

        print("[Info] Start processing videos...", flush=True)

        with torch.no_grad():
            video_embeds, video_id_arr = self._extract_video_embeds()
            self.start_retrieval_loop(video_embeds, video_id_arr)

    def start_retrieval_loop(self, video_embeds, video_id_arr):
        input_file = "input.txt"
        output_file = "output.txt"
        last_query = "a"
        print("[Info] Start retrieval. Waiting for input text queries...", flush=True)
        while True:
            time.sleep(1)
            with open(input_file, "r") as f:
                text = f.read().strip()
            if text == "":
                continue
            if text == last_query:
                continue

            print(f"[Info] Start processing query...", flush=True)
            last_query = text

            text_data = self._process_text([text])
            text_embeds = self.model.clip.get_text_features(**text_data)

            print(f"[Info] Query embed shape: {text_embeds.shape}", flush=True)
            print(f"[Info] Video embeds shape: {video_embeds.shape}", flush=True)

            # Pool frames for inference once we have all texts and videos
            video_embeds_pooled = self.model.pool_frames(text_embeds, video_embeds)
            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            sims = sims.squeeze(0)

            with open(output_file, "w") as f:
                f.write("Top 5 retrieved video IDs and sims:\n")
                for idx in sims.topk(5).indices:
                    vid = video_id_arr[idx]
                    sim = sims[idx].item()
                    f.write(f"video id: {vid}, sims: {sim:.4f}\n")
