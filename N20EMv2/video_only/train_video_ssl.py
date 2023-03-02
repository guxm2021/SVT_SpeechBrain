#!/usr/bin/env/python3
"""Recipe for training a encoder-based SVT system with N20EMv2 dataset
The system employs AV-HuBERT as its encoder. 
To run this recipe, do the following:
> python train_encoder.py hparams/train_encoder.yaml

Authors
 * Xiangming Gu 2022
"""

import os
import sys
import json
import torch
import logging
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from utils import frame2note, AverageMeter, Compose, Normalize, RandomCrop, HorizontalFlip, CenterCrop
from mir_eval import transcription, util
logger = logging.getLogger(__name__)


# Define training procedure
class SVT(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        video, video_lens = batch.sig
        video, video_lens = video.to(self.device), video_lens.to(self.device)

        # Forward pass
        video = video.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]
        video_feats = {"video": video, "audio": None}
        feats = self.modules.encoder(video_feats)   # (batch, frame, feat_dim)
        # Compute outputs
        pitch_octave_num = self.hparams.pitch_octave_num
        pitch_class_num = self.hparams.pitch_class_num

        logits = self.modules.head(feats)      # (batch, frame, 2+pitch_class+pitch_octave+2)
        # print(logits)
        onset_logits = logits[:, :, 0]
        offset_logits = logits[:, :, 1]
        pitch_out = logits[:, :, 2:]
        pitch_octave_logits = pitch_out[:, :, 0:pitch_octave_num+1]
        pitch_class_logits = pitch_out[:, :, pitch_octave_num+1:]

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits,  video_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        # predictions
        onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, video_lens = predictions
        
        # ground truth
        ids = batch.id
        anno, anno_lens = batch.anno
        anno, anno_lens = anno.to(self.device), anno_lens.to(self.device)
        onset_prob_gt = anno[:, :, 0].float()
        offset_prob_gt = anno[:, :, 1].float()
        pitch_octave_gt = anno[:, :, 2].long()
        pitch_class_gt = anno[:, :, 3].long()
        
        # Compute BCE Loss
        onset_positive_weight = torch.tensor([self.hparams.onset_positive_weight,], device=self.device)
        onset_loss = self.hparams.onset_criterion(onset_logits, onset_prob_gt, length=video_lens, pos_weight=onset_positive_weight)
        offset_positive_weight = torch.tensor([self.hparams.offset_positive_weight,], device=self.device)
        offset_loss = self.hparams.offset_criterion(offset_logits, offset_prob_gt, length=video_lens,pos_weight=offset_positive_weight)

        # Compute CE Loss
        pitch_octave_log_prob = self.hparams.log_softmax(pitch_octave_logits)
        octave_loss = self.hparams.octave_criterion(pitch_octave_log_prob, pitch_octave_gt, length=video_lens)
        pitch_class_log_prob = self.hparams.log_softmax(pitch_class_logits)
        pitch_loss = self.hparams.pitch_criterion(pitch_class_log_prob, pitch_class_gt, length=video_lens)

        # Compute Total Loss
        loss = onset_loss + offset_loss + octave_loss + pitch_loss

        if stage != sb.Stage.TRAIN:
            # Record loss terms
            self.onset_loss_metric.append(ids, onset_logits, onset_prob_gt, video_lens, None, onset_positive_weight)
            self.offset_loss_metric.append(ids, offset_logits, offset_prob_gt, video_lens, None, offset_positive_weight)
            self.octave_loss_metric.append(ids, pitch_octave_log_prob, pitch_octave_gt, video_lens)
            self.pitch_loss_metric.append(ids, pitch_class_log_prob, pitch_class_gt, video_lens)
            # Combine the predictions of multiple utterances and obtain the note-level prediction
            cur_utter = batch.cur_utter.item()
            all_utter = batch.all_utter.item()
            last_utter = self.last_utter
            assert cur_utter == last_utter + 1 or cur_utter == 1
            batch_size, frame = onset_logits.shape[:2]
            assert batch_size == 1   # assert batch_size = 1 during the evaluation
            
            # gpu -> cpu
            onset_probs, offset_probs = torch.sigmoid(onset_logits[0]).cpu(), torch.sigmoid(offset_logits[0]).cpu()
            pitch_octave_logits, pitch_class_logits = pitch_octave_logits[0].cpu(), pitch_class_logits[0].cpu()
            for f in range(frame):
                frame_info = (
                    onset_probs[f], offset_probs[f], torch.argmax(pitch_octave_logits[f]).item(),
                    torch.argmax(pitch_class_logits[f]).item()
                )
                self.song_pred.append(frame_info)
            if cur_utter == all_utter:
                # we reach the end of a song
                # estimation
                est_result = frame2note(self.song_pred, onset_thres=self.hparams.onset_threshold, 
                                        offset_thres=self.hparams.offset_threshold,
                                        frame_size=1/self.hparams.frame_rate)
                est_result_np = np.array(est_result)
                if est_result_np.shape[0] == 0:
                    print("There are no detected note events! All metrics for this song are set as zero!")
                    self.COnPOff_precis.update(0.0)
                    self.COnPOff_recall.update(0.0)
                    self.COnPOff_f1.update(0.0)
                    self.COnP_precis.update(0.0)
                    self.COnP_recall.update(0.0)
                    self.COnP_f1.update(0.0)
                    self.COn_precis.update(0.0)
                    self.COn_recall.update(0.0)
                    self.COn_f1.update(0.0)
                    self.COff_precis.update(0.0)
                    self.COff_recall.update(0.0)
                    self.COff_f1.update(0.0)
                else:
                    est_intervals = est_result_np[:, :2]
                    est_pitchs = est_result_np[:, 2]
                
                    # reference
                    ref_intervals, _ = batch.ref_intervals
                    ref_pitchs, _ = batch.ref_pitchs
                    ref_intervals = ref_intervals[0].cpu().numpy()
                    ref_pitchs = ref_pitchs[0].cpu().numpy()

                    ref_pitchs = util.midi_to_hz(ref_pitchs)
                    est_pitchs = util.midi_to_hz(est_pitchs)

                    # compute the metric
                    raw_data = transcription.evaluate(ref_intervals, ref_pitchs, est_intervals, est_pitchs, 
                                                      onset_tolerance=self.hparams.onset_tolerance,
                                                      offset_min_tolerance=self.hparams.offset_tolerance,
                                                      pitch_tolerance=self.hparams.pitch_tolerance)
                    self.COnPOff_precis.update(raw_data['Precision'])
                    self.COnPOff_recall.update(raw_data['Recall'])
                    self.COnPOff_f1.update(raw_data['F-measure'])
                    self.COnP_precis.update(raw_data['Precision_no_offset'])
                    self.COnP_recall.update(raw_data['Recall_no_offset'])
                    self.COnP_f1.update(raw_data['F-measure_no_offset'])
                    self.COn_precis.update(raw_data['Onset_Precision'])
                    self.COn_recall.update(raw_data['Onset_Recall'])
                    self.COn_f1.update(raw_data['Onset_F-measure'])
                    self.COff_precis.update(raw_data['Offset_Precision'])
                    self.COff_recall.update(raw_data['Offset_Recall'])
                    self.COff_f1.update(raw_data['Offset_F-measure'])
                # clear self.song_pred
                self.song_pred = []

            # update last_utter
            self.last_utter = cur_utter
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.encoder_optimizer.step()
            self.head_optimizer.step()

        self.encoder_optimizer.zero_grad()
        self.head_optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.onset_loss_metric = self.hparams.onset_stats()
        self.offset_loss_metric = self.hparams.offset_stats()
        self.octave_loss_metric = self.hparams.octave_stats()
        self.pitch_loss_metric = self.hparams.pitch_stats()
        if stage != sb.Stage.TRAIN:
            self.last_utter = 0    # determine the order of utterances
            self.song_pred = []    # record predictions of each song
            self.cur_epoch = epoch # record the current epoch
            self.COnPOff_precis = AverageMeter() # self.hparams.COnPOff_precis
            self.COnPOff_precis.reset()
            self.COnPOff_recall = AverageMeter() # self.hparams.OnPOff_recall
            self.COnPOff_recall.reset()
            self.COnPOff_f1 = AverageMeter() # self.hparams.COnPOff_f1
            self.COnPOff_f1.reset()
            self.COnP_precis = AverageMeter() # self.hparams.COnP_precis
            self.COnP_precis.reset()
            self.COnP_recall = AverageMeter() # self.hparams.COnP_recall
            self.COnP_recall.reset()
            self.COnP_f1 = AverageMeter() # self.hparams.COnP_f1
            self.COnP_f1.reset()
            self.COn_precis = AverageMeter() # self.hparams.COn_precis
            self.COn_precis.reset()
            self.COn_recall = AverageMeter() # self.hparams.COn_recall
            self.COn_recall.reset()
            self.COn_f1 = AverageMeter() # self.hparams.COn_f1
            self.COn_f1.reset()
            self.COff_precis = AverageMeter() # self.hparams.COff_precis
            self.COff_precis.reset()
            self.COff_recall = AverageMeter() # self.hparams.COff_recall
            self.COff_recall.reset()
            self.COff_f1 = AverageMeter() # self.hparams.COff_f1
            self.COff_f1.reset()
        else:
            # linear probing
            if epoch <= self.hparams.linear_prob_epochs:
                print("Stage for linear probing")
                self.set_requires_grad(self.modules.encoder, False)
            else:
                print("Stage for full finetuning")
                self.set_requires_grad(self.modules.encoder, True)
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """

        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )
        
        # save wav2vec 2.0 and encoder-decoder model
        if self.hparams.save_model:
            os.makedirs(self.hparams.save_model_folder, exist_ok=True)
            torch.save(self.modules.encoder.state_dict(), os.path.join(self.hparams.save_model_folder, 'avhubert.pt'))
            torch.save(self.modules.head.state_dict(), os.path.join(self.hparams.save_model_folder, 'video_cls.pt'))
            logger.info(f"Save wav2vec 2.0 and classifier to the folder: {self.hparams.save_model_folder}")
        else:
            logger.info("No wav2vec 2.0 and classifier to save")
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )
        
        if self.hparams.pretrain:
            logger.info(f"Load encoder model weights from the folder: {self.hparams.pretrain_folder}")
            self.modules.encoder.load_state_dict(torch.load(os.path.join(self.hparams.pretrain_folder, "encoder.pt")))
        else:
            logger.info("No pretrained encoder to be transferred")
            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["COnPOff_precis"] = self.COnPOff_precis.avg
            stage_stats["COnPOff_recall"] = self.COnPOff_recall.avg
            stage_stats["COnPOff_f1"] = self.COnPOff_f1.avg
            stage_stats["COnP_precis"] = self.COnP_precis.avg
            stage_stats["COnP_recall"] = self.COnP_recall.avg
            stage_stats["COnP_f1"] = self.COnP_f1.avg
            stage_stats["COn_precis"] = self.COn_precis.avg
            stage_stats["COn_recall"] = self.COn_recall.avg
            stage_stats["COn_f1"] = self.COn_f1.avg
            stage_stats["COff_precis"] = self.COff_precis.avg
            stage_stats["COff_recall"] = self.COff_recall.avg
            stage_stats["COff_f1"] = self.COff_f1.avg

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_head, new_lr_head = self.hparams.lr_annealing_head(
                stage_stats["loss"]
            )
            old_lr_encoder, new_lr_encoder = self.hparams.lr_annealing_encoder(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.head_optimizer, new_lr_head
            )
            sb.nnet.schedulers.update_learning_rate(
                self.encoder_optimizer, new_lr_encoder
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_head": old_lr_head,
                    "lr_encoder": old_lr_encoder,
                },
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                    "onset_loss": self.onset_loss_metric.summarize("average"),
                    "offset_loss": self.offset_loss_metric.summarize("average"),
                    "octave_loss": self.octave_loss_metric.summarize("average"),
                    "pitch_loss": self.pitch_loss_metric.summarize("average"),
                    "COnPOff_precis": stage_stats["COnPOff_precis"],
                    "COnPOff_recall": stage_stats["COnPOff_recall"],
                    "COnPOff_f1": stage_stats["COnPOff_f1"],
                    "COnP_precis": stage_stats["COnP_precis"],
                    "COnP_recall": stage_stats["COnP_recall"],
                    "COnP_f1": stage_stats["COnP_f1"],
                    "COn_precis": stage_stats["COn_precis"],
                    "COn_recall": stage_stats["COn_recall"],
                    "COn_f1": stage_stats["COn_f1"],
                    "COff_precis": stage_stats["COff_precis"],
                    "COff_recall": stage_stats["COff_recall"],
                    "COff_f1": stage_stats["COff_f1"],
                }
            )
            # self.checkpointer.save_and_keep_only(
            #     meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            # )
            self.checkpointer.save_and_keep_only(
                meta={"COnPOff_f1": stage_stats["COnPOff_f1"]}, max_keys=["COnPOff_f1"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                    "COnPOff_f1": stage_stats["COnPOff_f1"],
                    "COnP_f1": stage_stats["COnP_f1"],
                    "COn_f1": stage_stats["COn_f1"],
                    "COff_f1": stage_stats["COff_f1"],
                }
            )

    def init_optimizers(self):
        "Initializes the encoder optimizer and head optimizer"
        self.encoder_optimizer = self.hparams.encoder_opt_class(
            self.modules.encoder.parameters()
        )
        self.head_optimizer = self.hparams.head_opt_class(
            self.modules.head.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("encoder_opt", self.encoder_optimizer)
            self.checkpointer.add_recoverable("head_opt", self.head_optimizer)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # NOTE: cannot filter the data. keep the utterances of each song are together
    # valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        # NOTE: cannot filter the data. keep the utterances of each song are together
        # test_datasets[name] = test_datasets[name].filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    train_datasets = [train_data]
    eval_datasets = [valid_data] + [i for k, i in test_datasets.items()]
    
    dur_threshold = hparams["dur_threshold"]
    overlap = hparams["overlap"]
    sample_rate = hparams["sample_rate"]
    frame_rate = hparams["frame_rate"]
    split_noise = hparams["split_noise"]
    stride = dur_threshold - overlap
    # 2. Define video pipeline:
    image_crop_size = 88
    image_mean = 0.421
    image_std = 0.165
    transform_train = Compose([
                      Normalize( 0.0,255.0 ),
                      RandomCrop((image_crop_size, image_crop_size)),
                      HorizontalFlip(0.5),
                      Normalize(image_mean, image_std) ])

    transform_eval  = Compose([
                      Normalize( 0.0,255.0 ),
                      CenterCrop((image_crop_size, image_crop_size)),
                      Normalize(image_mean, image_std) ])
    # 2. Define audio and frame anno pipeline:
    @sb.utils.data_pipeline.takes("video", "duration", "frame_anno", "utter_id", "utter_num")
    @sb.utils.data_pipeline.provides("sig", "anno", "cur_utter", "all_utter")
    def utterance_train_pipeline(video, duration, frame_anno, utter_id, utter_num):
        # For training set, there are overlaps and random boundaries
        # 1. create random boundaries
        # original location: 0, random shift: -stride/2~+stride/2
        # if utter_id == 1, random shift: 0~stride/2
        # if utter_id == utter_num, and if duration < threshold, random shift: -stride/2~0, and if duration > threshold, random shift: 0~+stride/2
        if split_noise:
            shift = torch.rand(1).item() * stride - stride / 2
        else:
            shift = 0
        # load video
        sig = np.load(video) # (T, H, W)
        # data pre-processing
        sig = transform_train(sig)
        sig = np.expand_dims(sig, axis=-1)
        sig = torch.from_numpy(sig.astype(np.float32)) # (T, H, W, C)
        utter_id = int(utter_id)
        utter_num = int(utter_num)
        duration = float(duration)
        if utter_id == 1:
            # random shift: 0~stride/2
            num_sample1 = sample_rate * abs(shift)
            num_sample2 = sample_rate * abs(shift) + sample_rate * dur_threshold
            num_sample1 = round(num_sample1)
            num_sample2 = round(num_sample2)
            sig = sig[num_sample1:num_sample2]
        if utter_id == utter_num:
            # if duration < threshold, random shift: -stride/2~0, and if duration > threshold, random shift: 0~+stride/2
            if duration < dur_threshold:
                num_sample = (utter_id - 1) * sample_rate * stride - sample_rate * abs(shift)
            else:
                num_sample = (utter_id - 1) * sample_rate * stride + sample_rate * abs(shift)
            num_sample = round(num_sample)
            sig = sig[num_sample:]
        else:
            # random shift: -stride/2~+stride/2
            num_sample1 = (utter_id - 1) * sample_rate * stride + sample_rate * shift
            num_sample2 = (utter_id - 1) * sample_rate * stride + sample_rate * shift + sample_rate * dur_threshold
            num_sample1 = round(num_sample1)
            num_sample2 = round(num_sample2)
            sig = sig[num_sample1:num_sample2]
        # 3. load frame-level annotation
        anno = np.load(frame_anno)
        anno = torch.from_numpy(anno)
        if utter_id == 1:
            # random shift: 0~stride/2
            num_frame1 = frame_rate * abs(shift)
            num_frame2 = frame_rate * abs(shift) + frame_rate * dur_threshold
            num_frame1 = round(num_frame1)
            num_frame2 = round(num_frame2)
            anno = anno[num_frame1:num_frame2]
        elif utter_id == utter_num:
            # if duration < threshold, random shift: -stride/2~0, and if duration > threshold, random shift: 0~+stride/2
            if duration < dur_threshold:
                num_frame = (utter_id - 1) * frame_rate * stride - frame_rate * abs(shift)
            else:
                num_frame = (utter_id - 1) * frame_rate * stride + frame_rate * abs(shift)
            num_frame = round(num_frame)
            anno = anno[num_frame:]
        else:
            # random shift: -stride/2~+stride/2
            num_frame1 = (utter_id - 1) * frame_rate * stride + frame_rate * shift
            num_frame1 = round(num_frame1)
            num_frame2 = (utter_id - 1) * frame_rate * stride + frame_rate * shift + frame_rate * dur_threshold
            num_frame2 = round(num_frame2)
            anno = anno[num_frame1:num_frame2]
        cur_utter = utter_id
        all_utter = utter_num
        return sig, anno, cur_utter, all_utter
    
    @sb.utils.data_pipeline.takes("video", "frame_anno", "utter_id", "utter_num")
    @sb.utils.data_pipeline.provides("sig", "anno", "cur_utter", "all_utter")
    def utterance_eval_pipeline(video, frame_anno, utter_id, utter_num):
        # For evaluation sets, there are no overlaps, no random boundaries
        # load video
        sig = np.load(video) # (T, H, W)
        # data pre-processing
        sig = transform_eval(sig)
        sig = np.expand_dims(sig, axis=-1)
        sig = torch.from_numpy(sig.astype(np.float32)) # (T, H, W, C)
        utter_id = int(utter_id)
        utter_num = int(utter_num)
        if utter_id == utter_num:
            num_sample = (utter_id - 1) * sample_rate * dur_threshold
            num_sample = round(num_sample)
            sig = sig[num_sample:]
        else:
            num_sample1 = (utter_id - 1) * sample_rate * dur_threshold
            num_sample2 = utter_id * sample_rate * dur_threshold
            num_sample1 = round(num_sample1)
            num_sample2 = round(num_sample2)
            sig = sig[num_sample1:num_sample2]
        # 2. load frame-level annotations
        anno = np.load(frame_anno)
        anno = torch.from_numpy(anno)
        if utter_id == utter_num:
            num_frame = (utter_id - 1) * frame_rate * dur_threshold
            num_frame = round(num_frame)
            anno = anno[num_frame:]
        else:
            num_frame1 = (utter_id - 1) * frame_rate * dur_threshold
            num_frame1 = round(num_frame1)
            num_frame2 = utter_id * frame_rate * dur_threshold
            num_frame2 = round(num_frame2)
            anno = anno[num_frame1:num_frame2]
        cur_utter = utter_id
        all_utter = utter_num
        return sig, anno, cur_utter, all_utter
    
    sb.dataio.dataset.add_dynamic_item(train_datasets, utterance_train_pipeline)
    sb.dataio.dataset.add_dynamic_item(eval_datasets, utterance_eval_pipeline)
    
    # 3. Define annotation pipeline:
    @sb.utils.data_pipeline.takes("song_anno")
    @sb.utils.data_pipeline.provides("ref_intervals", "ref_pitchs")
    def song_pipeline(song_anno):
        # handle the note-level annotations
        with open(song_anno) as json_data:
            song_anno = json.load(json_data)
        json_data.close()
        song_anno_np = np.array(song_anno)
        ref_interval_np = song_anno_np[:, :2]
        ref_pitchs_np = song_anno_np[:, 2]
        ref_intervals = torch.from_numpy(ref_interval_np)
        ref_pitchs = torch.from_numpy(ref_pitchs_np)
        return ref_intervals, ref_pitchs

    sb.dataio.dataset.add_dynamic_item(datasets, song_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "anno", "cur_utter", "all_utter", "ref_intervals", "ref_pitchs"],
    )
    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = SVT(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    # for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #     asr_brain.evaluate(
    #         test_datasets[k], 
    #         min_key="loss",        # changed by Xiangming, use min_key "loss" to load checkpointer
    #         test_loader_kwargs=hparams["test_dataloader_opts"]
    #     )
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.evaluate(
            test_datasets[k], 
            max_key="COnPOff_f1",
            test_loader_kwargs=hparams["test_dataloader_opts"]
        )
