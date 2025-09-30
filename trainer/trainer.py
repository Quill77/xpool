from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, compute_accuracy
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _process_text(self, text):
        if self.tokenizer is not None:
            text = self.tokenizer(text, return_tensors='pt', padding=True,
                                  truncation=True)
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        else:
            text = {key: val.to(self.device) for key, val in text.items()}
        return text
            
    
    def _log(self, epoch, batch_idx, num_steps, loss_val):
        print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss_val))
    
    def _eval(self, epoch, batch_idx, num_steps):
        val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
        self.model.train()

        if val_res['R1-window'] > self.best_window:
            self.best_window = val_res['R1-window']
            self._save_checkpoint(epoch, save_best=True)

        if val_res['R1'] > self.best:
            self.best = val_res['R1']

        print(" Current Best Window Average R@1 is {}".format(self.best_window))
        print(" Current Best R@1 is {}\n\n".format(self.best))
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            data['text'] = self._process_text(data['text'])
            data['video'] = data['video'].to(self.device)

            text_embeds, video_embeds, video_embeds_pooled = self.model(data)
            output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            

            loss = self.loss(output, self.model.clip.logit_scale)
            loss.backward()
            loss_val = loss.detach().item()
            total_loss += loss_val
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss_val, self.global_step)


            if batch_idx % self.log_step == 0:
                self._log(epoch, batch_idx, num_steps, loss_val)

            if batch_idx in eval_steps:
                self._eval(epoch, batch_idx, num_steps)

        res = {
            'loss_train':  total_loss / num_steps
        }

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
        video_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                data['text'] = self._process_text(data['text'])
                data['video'] = data['video'].to(self.device)
                
                text_embed, video_embed, video_embed_pooled = self.model(data)
                text_embed_arr.append(text_embed.cpu())
                video_embed_arr.append(video_embed.cpu())
                sims_batch = sim_matrix_training(text_embed, video_embed_pooled, self.pooling_type)

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    video_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            video_embeds = torch.cat(video_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            all_video_ids_set = set(video_ids)
            video_embeds_per_video_id = {video_id: video_embeds[idx] for idx, video_id in enumerate(all_video_ids_set)}

                
            video_embeds = torch.stack(list(video_embeds_per_video_id.values()))
             
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            video_embeds_pooled = self.model.pool_frames(text_embeds, video_embeds)
            self.model.pool_frames.cuda()

            text_embeds_per_video_id, video_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, 
                    video_embeds_pooled, video_ids, self.pooling_type)
            
            sims = sim_matrix_inference(text_embeds_per_video_id, video_embeds_pooled_per_video_id, self.pooling_type)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res

    
    

    def _train_suscape(self, epoch):
        """
        TODO: modify docstring
        Special training logic for Suscape dataset
        Note:
            For Suscape test split, we have multiple captions per video.
            During training, we treat each caption as a separate text-video pair.
            During validation, we first get all text and video embeddings, then pool video frames once
            to get video embeddings, and finally average on all captions belonging to the same video.
        :return: A log that contains all information you want to save.
        """
        # if config.dataset_name == 'suscape_long':
        #     descs = self.train_data_loader.dataset.descs
        #     self.suscape_text_embeds = self.tokenizer(descs, return_tensors='pt', padding=True, truncation=True)
        #     self.suscape_text_embeds = {key: val.to(self.device) for key, val in self.suscape_text_embeds.items()}
        #     print("suscape text embeds:")
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        

        for batch_idx, data in enumerate(self.train_data_loader):
            data['text'] = self.suscape_text_embeds
            data['video'] = data['video'].to(self.device)
            
            text_embeds, video_embeds_pooled = self.model(data)
            
            output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            
            # Assuming the labels are provided in the data dictionary
            labels = data['labels'].T.to(self.device)
            
            loss = self.loss(output, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                self._test_suscape()
                self.model.train()
                self._save_checkpoint(epoch, save_best=True)

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res
    
    def _test_suscape(self):
        """
        Special testing logic for Suscape dataset
        Note:
            For Suscape test split, we have multiple captions per video.
            During testing, we first get all text and video embeddings, then pool video frames once
            to get video embeddings, and finally average on all captions belonging to the same video.
        :return: A log that contains all information you want to save.
        """
        self.model.eval()
        total_test_loss = 0.0
        text_embed_arr = []
        video_embed_arr = []
        all_video_ids = []
        all_labels = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                data['text'] = self.suscape_text_embeds
                data['video'] = data['video'].to(self.device)
                data['labels'] = data['labels'].to(self.device)

                text_embed, video_embed, video_embed_pooled = self.model(data, return_all_frames=True)
                text_embed_arr.append(text_embed)
                video_embed_arr.append(video_embed)
                sims_batch = sim_matrix_training(text_embed, video_embed_pooled, self.pooling_type)
                
                print(sims_batch[0])
                print(data['labels'][0])
                curr_loss = self.loss(sims_batch, data['labels'].T)
                
                
                accuracy = compute_accuracy(sims_batch, data['labels'])
                print(f"Batch Accuracy: {accuracy}")
                
                total_test_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_video_ids.append(v_id)
                all_labels.append(data['labels'])

            text_embeds = torch.cat(text_embed_arr)
            video_embeds = torch.cat(video_embed_arr)
            all_labels = torch.cat(all_labels)

            # Pool frames for inference once we have all texts and videos
            video_embeds_pooled = self.model.pool_frames(text_embeds, video_embeds)

            accuracy = compute_accuracy(sims_batch, data['labels'])

            total_test_loss = total_test_loss / len(self.valid_data_loader)

            # metrics = self.metrics
            # res = metrics(sims, all_labels)
            
            print(f"-----Test Results-----\n",
                # f"R@1: {res['R1']}\n", 
                # f"R@5: {res['R5']}\n", 
                # f"R@10: {res['R10']}\n",
                # f"MedR: {res['MedR']}\n",
                # f"MeanR: {res['MeanR']}\n",
                f"Accuracy: {accuracy}\n",
                f"Loss: {total_test_loss}")
            
            # res['loss_test'] = total_test_loss

            # if self.writer is not None:
            #     for m in res:
            #         self.writer.add_scalar(f'test/{m}', res[m], self.global_step)

            # return res
        
