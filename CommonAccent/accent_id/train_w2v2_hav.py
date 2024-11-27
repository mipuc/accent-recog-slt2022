#!/usr/bin/env python3
import logging
import os
import sys
import speechbrain as sb
import torch
import torchaudio
import librosa
from common_accent_prepare import prepare_common_accent
from hyperpyyaml import load_hyperpyyaml
import pickle
import numpy as np
import math
import pandas as pd
logger = logging.getLogger(__name__)

import ipdb

# Brain class for Accent ID training
class AID(sb.Brain):
    def __init__(self, gemeinde_df, accents_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemeinde_df = gemeinde_df
        self.accents_encoder = accents_encoder

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, wav_lens = wavs
        
        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            # added the False for now, to avoid augmentation of any type
            wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens], dim=0)
        
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        
        # Feature extraction and normalization
        # wavs = self.modules.mean_var_norm_input(wavs, wav_lens)       

        # forward pass HF (possible: pre-trained) model
        # feats = self.modules.wav2vec2(wavs, wav_lens=wav_lens)
        feats = self.modules.wav2vec2(wavs)

        return feats, wav_lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)

        # last dim will be used for pooling, 
        # StatisticsPooling uses 'lens'
        if hparams["avg_pool_class"] == "statpool":
            outputs = self.hparams.avg_pool(feats, lens)
        elif hparams["avg_pool_class"] == "avgpool":
            outputs = self.hparams.avg_pool(feats)
            # this uses a kernel, thus the output dim is not 1 (mean to reduce)
            outputs = outputs.mean(dim=1)
        else:
            outputs = self.hparams.avg_pool(feats)

        # ipdb.set_trace()
        # preparing outputs
        outputs = outputs.view(outputs.shape[0], -1)
        embeddings = outputs
        # print(self.modules)
        # outputs = self.modules.preout_mlp(outputs)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs, lens, embeddings

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens, embeddings = inputs

        # get the targets from the batch
        targets = batch.accent_encoded.data

        # to meet the input form of nll loss
        targets = targets.squeeze(1)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            targets = torch.cat([targets, targets], dim=0)
            lens = torch.cat([lens, lens], dim=0)

            # if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            #     self.hparams.lr_annealing.on_batch_end(self.optimizer)
        
        # save embeddings
        export_embeddings = False
        if export_embeddings:
            self.save_embeddings_and_labels(embeddings, targets, stage)
        
        # get the final loss
        loss = self.hparams.compute_cost(predictions, targets)
        
        # Get class coordinates (longitude and latitude)
        # get encoded accents and map to accents
        # accents_encoder_test = sb.dataio.encoder.CategoricalEncoder()
        # accents_encoder_test.load(path=os.path.join(hparams["save_folder"], "accent_encoder.txt"))
        accents_batch_targets = self.accents_encoder.decode_torch(targets)
        accents_batch_predictions = self.accents_encoder.decode_torch(predictions.argmax(dim=1))
        # map accents to coordinates
        # gemeinde_csv = "/home/projects/vokquant/accent-recog-slt2022/CommonAccent/data/gemeinde_coordinates.csv"
        # gemeinde_df = pd.read_csv(gemeinde_csv)

        # Calculate distance penalty
        # print("loss: ", loss)
        for i in range(len(targets)):
            true_class = targets[i].item()
            predicted_class = predictions.argmax(dim=1)[i].item()
            
            if true_class != predicted_class:
                lat1 = self.gemeinde_df[self.gemeinde_df['Gemeindekennzahl'] == int(accents_batch_targets[i])]['Latitude'].values[0]
                lon1 = self.gemeinde_df[self.gemeinde_df['Gemeindekennzahl'] == int(accents_batch_targets[i])]['Longitude'].values[0]
                lat2 = self.gemeinde_df[self.gemeinde_df['Gemeindekennzahl'] == int(accents_batch_predictions[i])]['Latitude'].values[0]
                lon2 = self.gemeinde_df[self.gemeinde_df['Gemeindekennzahl'] == int(accents_batch_predictions[i])]['Longitude'].values[0]
                distance = haversine(lat1, lon1, lat2, lon2)
                
                # Adding distance to the loss
                # distance_factor = hparams["distance_penalty_weight"]  # You can scale the distance as needed
                distance_factor = 0.02
                loss += distance * distance_factor
        # print("loss after distance penalty: ", loss)
        # ipdb.set_trace()
        # append the metrics for evaluation
        if stage != sb.Stage.TRAIN:
            # ipdb.set_trace()
            self.error_metrics.append(batch.id, predictions, targets)
            self.error_metrics2.append(batch.id, predictions.argmax(-1), targets)
            
            # compute the accuracy of the one-step-forward prediction
            # self.acc_metric.append(predictions, targets, lens)
            self.acc_metric.append(predictions, targets.view(1, -1), lens)
            self.acc_metric2.append(predictions.argmax(-1), targets.view(1, -1), lens)
        
        export_predictions = False
        if export_predictions:
            self.save_predictions(batch.id, predictions, targets, lens)
        
        return loss

    def save_predictions(self, batch_ids, predictions, targets, lens):
        # Create a dictionary to store results
        results = {
            "batch_ids": batch_ids,
            "predictions": predictions.cpu().detach().numpy(),
            "targets": targets.cpu().detach().numpy(),
            "lens": lens.cpu().detach().numpy(),
        }
        
        # Define where to save the results
        save_path = os.path.join(self.hparams.save_folder, "predictions", "test_predictions.pkl")
        # os.remove(save_path) if os.path.exists(save_path) else None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save results as a pickle file
        with open(save_path, "ab") as f:
            pickle.dump(results, f)
    
    
    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        should_step = self.step % self.grad_accumulation_factor == 0

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        with self.no_sync(not should_step):
            (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                self.wav2vec2_optimizer.step()
                self.optimizer.step()
            self.wav2vec2_optimizer.zero_grad()
            self.optimizer.zero_grad()
            self.optimizer_step += 1

        self.on_fit_batch_end(batch, predictions[0:2], loss, should_step)
        return loss.detach().cpu()

    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.acc_metric = self.hparams.acc_computer()
            self.error_metrics2 = self.hparams.error_stats()
            self.acc_metric2 = self.hparams.acc_computer()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            # self.train_stats = stage_stats
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["error_rate"] = self.error_metrics.summarize("average")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # ipdb.set_trace()
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            steps = self.optimizer_step

            # The train_logger writes a summary to stdout and to the logfile.
            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "wave2vec_lr": old_lr_wav2vec2,
                "steps": steps,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats={"loss": self.train_loss},
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none)

    def save_embeddings_and_labels(self, embeddings, labels, stage):
        """Saves embeddings and labels to a file for later analysis."""
        if stage == sb.Stage.TEST:
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            save_dir = hparams["save_folder"]
            save_dir = os.path.join(save_dir, "embeddings")
            os.makedirs(save_dir, exist_ok=True)
            # Define a file name based on stage
            filename = f"{save_dir}/embeddings_{stage}.pkl"
            
            # remove batch size dimension and append to pickle file one by one
            embeddings_np_list = []
            for i in range(embeddings_np.shape[0]):
                id = labels_np[i]
                embedding = embeddings_np[i]
                embeddings_np_list.append((id, embedding))
            
            # append to pickle file
            with open(filename, 'ab') as f:
                pickle.dump(embeddings_np_list, f)
    
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points 
    on the Earth (specified in decimal degrees)"""
    
    # haversine formula
    R = 6371.0  # radius of the Earth in km
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(math.radians(dlat) / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(math.radians(dlon) / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
    
def dataio_prep(hparams):
    # 1. Define train/valid/test datasets
    data_folder = hparams["csv_prepared_folder"]
    train_csv = os.path.join(data_folder, "train" + ".csv")
    valid_csv = os.path.join(data_folder, "dev" + ".csv")
    test_csv = os.path.join(data_folder, "test" + ".csv")

    # train_csv = os.path.join(data_folder, "dev" + ".csv")
    # valid_csv = os.path.join(data_folder, "test" + ".csv")

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv, replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv, replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv, replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'accent01': 0, 'accent02': 1, ..)
    accent_encoder = sb.dataio.encoder.CategoricalEncoder()
    
    @sb.utils.data_pipeline.takes("wav","duration","offset")
    @sb.utils.data_pipeline.provides("sig")
    def audio_offset_pipeline(wav,duration,offset):      
        sig, sr = librosa.load(wav,  sr=hparams["sample_rate"], offset=int(offset), duration=10)
        sig = torch.tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_offset_pipeline)

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("accent")
    @sb.utils.data_pipeline.provides("accent", "accent_encoded")
    def label_pipeline(accent):
        yield accent
        accent_encoded = accent_encoder.encode_label_torch(accent)
        yield accent_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "accent_encoded"],
    )

    accent_encoder_file = os.path.join(hparams["save_folder"], "accent_encoder.txt")
    accent_encoder.load_or_create(
        path=accent_encoder_file,
        from_didatasets=[train_data],
        output_key="accent",
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
        accent_encoder
    )

def get_pooling_layer(hparams):
    """function to get the pooling layer based on value in hparams file or CLI"""
    pooling = hparams["avg_pool_class"]
    
    # possible classes are statpool, adaptivepool, avgpool
    if pooling == "statpool":
        from speechbrain.nnet.pooling import StatisticsPooling
        pooling_layer = StatisticsPooling(return_std=False)
    elif pooling == "adaptivepool":
        from speechbrain.nnet.pooling import AdaptivePool
        pooling_layer = AdaptivePool(output_size=1)
    elif pooling == "avgpool":
        from speechbrain.nnet.pooling import Pooling1d
        pooling_layer = Pooling1d(pool_type="avg", kernel_size=3)
    else:
        raise ValueError("Pooling strategy must be in ['statpool', 'adaptivepool', 'avgpool']")
    hparams["avg_pool"] = pooling_layer

    return hparams

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_accent,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )
        
    # defining the Pooling strategy based on hparams file:
    hparams = get_pooling_layer(hparams)

    # Create dataset objects "train", "valid", and "test", train/val samples and accent_encoder
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
        accent_encoder
    ) = dataio_prep(hparams)

    # Load the Wav2Vec 2.0 model
    hparams["wav2vec2"] = hparams["wav2vec2"].to("cuda:0")
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()
    
    if hparams["load_pretrained"]:
        sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])
        print("Pretrained model loaded")
        print(sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files))
    else:
        print("No pretrained model loaded")
    ##
    gemeinde_csv = "/home/projects/vokquant/accent-recog-slt2022/CommonAccent/data/gemeinde_coordinates.csv"
    gemeinde_df = pd.read_csv(gemeinde_csv)
    
    # Load accents encoder
    accents_encoder = sb.dataio.encoder.CategoricalEncoder()
    accents_encoder.load(path=os.path.join(hparams["save_folder"], "accent_encoder.txt"))
    ##
    # Initialize the Brain object to prepare for mask training.
    aid_brain = AID(
        gemeinde_df=gemeinde_df,
        accents_encoder=accent_encoder,
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    aid_brain.fit(
        aid_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    
    save_dir = hparams["save_folder"]
    stage = "Stage.TEST"
    filename = f"{save_dir}/embeddings/embeddings_{stage}.pkl"
    print(f"filename for saving test pickle file {filename}")
    if os.path.exists(filename):
        print("Removing embeddings pickle file")
        os.remove(filename)
    else:
        print("The test pickle file does not exist, writing to file: ", filename)

    # Load the best checkpoint for evaluation
    test_stats = aid_brain.evaluate(
        test_set=valid_data,
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # validate for valid data (to export embeddings)
    
    # valid_stats = aid_brain.evaluate(
    #     test_set=valid_data,
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["valid_dataloader_opts"],
    # )
    # accents_lists_int=range(int(hparams["n_accents"]))
    # accents_list=[]

    # for a in accents_lists_int:
    #     accents_list.append(str(a))
    
    # print("Test for all accents")
    # print("accents_list: ", accents_list)
    # #get available accents in test_data using filtered_sorted
    # unique_accents = set([data['accent_encoded'].item() for data in test_data])
    # for acc in accents_list:
    #     if int(acc) in unique_accents:
    #         test_data_acc = test_data.filtered_sorted(key_test={"accent_encoded": lambda x: x.item() == int(acc)})
    #         # ipdb.set_trace()
    #         print("test_data_acc: ", test_data_acc)
    #         # get length of test_data_acc
    #         print("len(test_data_acc): ", len(test_data_acc))

    #         print("Test for: "+acc)
            
    #         test_stats = aid_brain.evaluate(
    #             test_set=test_data_acc,
    #             min_key="error_rate",
    #             test_loader_kwargs=hparams["test_dataloader_opts"],
    #         )