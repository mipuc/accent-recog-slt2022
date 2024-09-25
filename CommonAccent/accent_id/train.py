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

"""Recipe for training an Accent Classification system with CommonVoice Accent.

To run this recipe, do the following:
> python train.py hparams/train_ecapa_tdnn.yaml

Author
------
 * Juan Pablo Zuluaga 2023
"""

logger = logging.getLogger(__name__)

# Brain class for Accent ID training
class AID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            # added the False for now, to avoid augmentation of any type
            wavs_noise = self.modules.env_corrupt(wavs, lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            lens = torch.cat([lens, lens], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

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
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

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

        predictions, lens = inputs

        targets = batch.accent_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            targets = torch.cat([targets, targets], dim=0)
            lens = torch.cat([lens, lens], dim=0)

            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # get the final loss
        loss = self.hparams.compute_cost(predictions, targets, lens)

        # append the metrics for evaluation
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

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

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `common_accent_prepare` to have been called before this,
    so that the `train.csv`, `valid.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    
    # 1. Define train/valid/test datasets
    data_folder = hparams["csv_prepared_folder"]
    train_csv = os.path.join(data_folder, "train" + ".csv")
    valid_csv = os.path.join(data_folder, "dev" + ".csv")
    test_csv = os.path.join(data_folder, "test" + ".csv")

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv, replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            #key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            #key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            #key_max_value={"duration": hparams["avoid_if_longer_than"]},
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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        # sig, _ = torchaudio.load(wav)
        # sig = sig.transpose(0, 1).squeeze(1)        
        sig, _ = librosa.load(wav,  sr=hparams["sample_rate"], offset=0.0, duration=hparams["avoid_if_longer_than"])
        #sig, _ = librosa.load(wav, sr=hparams["sample_rate"])
        sig = torch.tensor(sig)
        return sig

    @sb.utils.data_pipeline.takes("wav","duration","offset")
    @sb.utils.data_pipeline.provides("sig")
    def audio_offset_pipeline(wav,duration,offset):      

        #logger.info("wav: "+str(wav))
        #logger.info("offset: "+str(offset))
        #logger.info("duration : "+str(duration))

        if float(duration)<float(offset)+10:
            logger.info("Too long: "+str(wav))
            exit(1)
        sig, sr = librosa.load(wav, sr=None, offset=int(offset), duration=10)

        #sig1, sr1 = librosa.load(wav, sr=None)
        #duri = librosa.get_duration(y=sig1, sr=sr1)

        #logger.info("duration1 : "+str(duri))

        sig = torch.tensor(sig)
        return sig

    #sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

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

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
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

    # Create dataset objects "train", "valid", and "test", train/val samples and accent_encoder
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
        accent_encoder
    ) = dataio_prep(hparams)

    # Fetch and laod pretrained moduls if True
    if hparams["load_pretrained"]:
        sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    aid_brain = AID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    print(aid_brain.modules)
    #PUM freeze all parameters
    if int(hparams["freeze_parameters"])==1:
        for param in aid_brain.modules.parameters():
            param.requires_grad = False
        for param in aid_brain.modules["classifier"].parameters():
            param.requires_grad = True
        for param in aid_brain.modules.parameters():
            print(str(param))    

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

    # Load the best checkpoint for evaluation
    test_stats = aid_brain.evaluate(
        test_set=test_data,
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    #accents_list=["african","australia","canada","england","indian","ireland","newzealand","philippines","scotland","singapore","us","wales"]
    #accents_list=["other","indian","ireland","newzealand","scotland","wales"]
    accents_lists_int=range(int(hparams["n_accents"]))
    accents_list=[]

    for a in accents_lists_int:
        accents_list.append(str(a))

    for acc in accents_list:

        test_data_acc = test_data.filtered_sorted(key_test={"accent": lambda x: x == acc})

        print("Test for "+acc)
        test_stats = aid_brain.evaluate(
            test_set=test_data_acc,
            min_key="error_rate",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )


