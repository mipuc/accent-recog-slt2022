import logging
import os
import sys

import librosa
import speechbrain as sb
import torch
import torchaudio
from common_accent_prepare import prepare_common_accent
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
import pickle
import datetime
import ipdb

"""Recipe for performing inference on Accent Classification system with CommonVoice Accent.

To run this recipe, do the following:
> python inference.py hparams/inference_ecapa_tdnn.yaml

Author
------
 * Juan Pablo Zuluaga 2023
"""

logger = logging.getLogger(__name__)


# Brain class for Accent ID training
class AccID_inf(sb.Brain):
    def eval(self):
        for module in self.modules.values():
            if isinstance(module, torch.nn.Module):
                module.eval()
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

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens.float())
        # print("shape(feats inside prepare): ", feats.shape)

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
        # print("shape(embeddings): ", embeddings.shape)
        
        outputs = self.modules.classifier(embeddings)
        
        export_embeddings = True
        if export_embeddings:
            # Save embeddings for t-SNE
            output_folder = "/home/projects/vokquant/accent-recog-slt2022/CommonAccent/results/ECAPA-TDNN/AT/"
            os.makedirs(output_folder+"/embeddings", exist_ok=True)
            save_path = os.path.join(output_folder, "embeddings")
            os.makedirs(save_path, exist_ok=True)
            # print("save_path: ", save_path)
            for i in range(len(batch.id)):
                # Get wav file name
                filename = batch.id[i]
                # Save embedding with filename
                torch.save(embeddings[i], os.path.join(save_path, filename + ".pt"))
        
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
        loss = self.hparams.compute_cost(predictions, targets)

        # Append the outputs here, we can access then later
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)
            self.error_metrics2.append(batch.id, predictions.argmax(dim=-1), targets)

        return loss

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

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.error_metrics2 = self.hparams.error_stats2()

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `common_accent_prepare` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    # @sb.utils.data_pipeline.takes("wav")
    # @sb.utils.data_pipeline.provides("sig")
    # def audio_pipeline(wav):
    #     """Load the signal, and pass it and its length to the corruption class.
    #     This is done on the CPU in the `collate_fn`."""
    #     # sig, _ = torchaudio.load(wav)
    #     # sig = sig.transpose(0, 1).squeeze(1)
    #     # Problem with Torchaudio while reading MP3 files (CommonVoice)
    #     sig, _ = librosa.load(wav, sr=hparams["sample_rate"])
    #     sig = torch.tensor(sig)
    #     return sig
    @sb.utils.data_pipeline.takes("wav","duration","offset")
    @sb.utils.data_pipeline.provides("sig")
    def audio_offset_pipeline(wav,duration,offset):      
        sig, sr = librosa.load(wav,  sr=hparams["sample_rate"], offset=int(offset), duration=10)
        sig = torch.tensor(sig)
        return sig

    # sb.dataio.dataset.add_dynamic_item(datasets, audio_offset_pipeline)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("accent")
    @sb.utils.data_pipeline.provides("accent", "accent_encoded")
    def label_pipeline(accent):
        yield accent
        accent_encoded = accent_encoder.encode_label_torch(accent)
        yield accent_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["csv_prepared_folder"], dataset + ".csv"),
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_offset_pipeline, label_pipeline],
            output_keys=["id", "sig", "accent_encoded", "duration"],
        )
        # filtering out recordings with more than max_audio_length allowed
        datasets[dataset] = datasets[dataset].filtered_sorted(
            key_max_value={"duration": hparams["max_audio_length"]},
        )

    return datasets

def return_embedding(file_path, accid_brain, save_path):
    # set batch size to 1
    # accid_brain.hparams.batch_size = 1
    # Load the audio file
    _ , sample_rate = torchaudio.load(file_path)
    sig, _ = librosa.load(file_path, sr=sample_rate)
    sig = torch.tensor(sig).unsqueeze(0)  # Add an extra dimension
    sig = sig.cuda()
    # Prepare the features
    feats, lens = accid_brain.prepare_features((sig, torch.tensor([sig.shape[0]])), sb.Stage.TEST)
    # print("shape(feats): ", feats.shape)
    # print("lens: ", lens)
    # Compute the embeddings
    embeddings = accid_brain.modules.embedding_model(feats)

    # Get the filename without the extension
    # print("file_path: ", file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # print("filename: ", filename)
    # Save the embedding with the same filename
    # torch.save(embeddings, os.path.join(save_path, filename + ".pt"))
    # print("Embedding saved at: ", os.path.join(save_path, filename + ".pt"))
    return embeddings

# map indices to encoder
def map_indices_to_encoder(indices, encoder):
    city_codes = [city_codes[i] for i in indices]
    return city_codes

def calc_macro_results(y_pred, y_true):
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)
    # calculate macro scores
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print("f1_macro: ", f1_macro)
    accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
    print("accuracy: ", accuracy)
    # precision = precision_score(y_true, y_pred, average='macro')
    # print("precision: ", precision)
    # recall = recall_score(y_true, y_pred, average='macro')
    # print("recall: ", recall)
    return {"f1_macro": f1_macro, "accuracy": accuracy}


def return_embedding(file_path, accid_brain, save_path):
    # set batch size to 1
    # accid_brain.hparams.batch_size = 1
    # Load the audio file
    _ , sample_rate = torchaudio.load(file_path)
    sig, _ = librosa.load(file_path, sr=sample_rate)
    sig = torch.tensor(sig).unsqueeze(0)  # Add an extra dimension
    sig = sig.cuda()
    # Prepare the features
    feats, lens = accid_brain.prepare_features((sig, torch.tensor([sig.shape[0]])), sb.Stage.TEST)
    # print("shape(feats): ", feats.shape)
    # print("lens: ", lens)
    # Compute the embeddings
    embeddings = accid_brain.modules.embedding_model(feats)

    # Get the filename without the extension
    # print("file_path: ", file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # print("filename: ", filename)
    # Save the embedding with the same filename
    # torch.save(embeddings, os.path.join(save_path, filename + ".pt"))
    # print("Embedding saved at: ", os.path.join(save_path, filename + ".pt"))
    return embeddings

def sort_by_states(accent_encoder):
    city_codes_ids = {}
    with open(accent_encoder, "r") as f:
        lines = f.readlines()
        for line in lines[:-2]:
            city_code = line.split(' => ')[0]
            city_code = city_code.replace("'", "")
            accent_id = line.split(' => ')[1].strip()
            city_codes_ids[city_code] = accent_id
    regions = [1, 2, 3, 4, 5, 6, 7, 8]
    region_ids = []
    region_city_codes = []
    for region in regions:
        region_cities = [city for city in city_codes_ids.keys() if city.startswith(str(region))]
        region_cities_ids = [city_codes_ids[city] for city in region_cities]
        region_ids.append(region_cities_ids)
        region_city_codes.append(region_cities)
    
    # check for errors in accent encoder:
    for k in range(len(region_ids)):
        tmp = []
        for i in range(len(region_ids[k])):
            tmp.append(region_city_codes[k][i][0])
            # check if all tmp values are the same 
            if len(set(tmp)) == 1:
                continue
            else:
                print("!!!Not all values are the same. Problem in accent_encoder!!!")
    
    return region_ids, region_city_codes

# Recipe begins!
if __name__ == "__main__":

    export_embeddings = False
    if export_embeddings:
        save_path = os.path.join(output_folder, "embeddings")
        # file_path = "/home/lorenzg/.cache/huggingface/datasets/downloads/extracted/7fa940d5e98e14130c923c2dde203c1729e46a596f788672e3580c0dece79a25/de_train_0/common_voice_de_27341092.mp3"
        # output_folder = "/home/projects/vokquant/accent-recog-slt2022/results/ECAPA-TDNN/AT/spkrec-ecapa-voxceleb/1987/"
    
    # this prevents to load an old model
    auto_rename = True
    if os.path.exists("./model_checkpoints"):
        print("WARNING: model_checkpoints directory exists. Please remove it before running inference.py")
        if auto_rename:
            print("automatic renaming of model_checkpoints directory")
            datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.rename("./model_checkpoints", f"./model_checkpoints_{datetime}")
    
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # print all models in speechbrain.lobes.models
    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create output directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g., 'accent01': 0, 'accent02': 1, ..)
    accent_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Load label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    accent_encoder_file = os.path.join(hparams["pretrained_path"], "accent_encoder.txt")
    accent_encoder.load_or_create(
        path=accent_encoder_file,
        output_key="accent",
    )

    # Create dataset objects "train", "dev", and "test" and accent_encoder
    datasets = dataio_prep(hparams)

    # Fetch and load pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is not available")
    
    hparams["pretrainer"].load_collected(device=run_opts["device"])
    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is not available")
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hparams["pretrainer"].load_collected(device=device)


    # Initialize the Brain object to prepare for performing inference.
    accid_brain = AccID_inf(
        modules=hparams["modules"],
        hparams=hparams,
    )
    accid_brain.eval()
    
    if export_embeddings:
        return_embedding(file_path, accid_brain, save_path)

    # Function that actually prints the output. you can modify this to get some other information
    def print_confusion_matrix(AccID_object, set_name="dev"):
        """pass the object what contains the stats"""

        # get the scores after running the forward pass
        y_true_val = torch.cat(AccID_object.error_metrics2.labels).tolist()
        y_pred_val = torch.cat(AccID_object.error_metrics2.scores).tolist()
        # print("y_pred_val: ", y_pred_val)
        # print("y_true_val: ", y_true_val)

        # with open("/home/projects/vokquant/TCS/scripts/region_ids.pkl", "rb") as f:
        #     region_ids = pickle.load(f)
        
        region_ids, region_city_codes = sort_by_states(accent_encoder_file)
        
        if region_city_codes[0][0][0] == region_city_codes[0][-1][0]:
            print("region_city_codes are probably correctly sorted by states")
        
        # ipdb.set_trace()
        region_1 = region_ids[0]
        region_2 = region_ids[1]
        region_3 = region_ids[2]
        region_4 = region_ids[3]
        region_5 = region_ids[4]
        region_6 = region_ids[5]
        region_7 = region_ids[6]
        region_8 = region_ids[7]
        
        # map y_true_val to region_ids
        y_true_val_mapped = []
        for i in y_true_val:
            i_str = str(i)  # Convert to string
            if i_str in region_1:
                y_true_val_mapped.append(0)
            elif i_str in region_2:
                y_true_val_mapped.append(1)
            elif i_str in region_3:
                y_true_val_mapped.append(2)
            elif i_str in region_4:
                y_true_val_mapped.append(3)
            elif i_str in region_5:
                y_true_val_mapped.append(4)
            elif i_str in region_6:
                y_true_val_mapped.append(5)
            elif i_str in region_7:
                y_true_val_mapped.append(6)
            elif i_str in region_8:
                y_true_val_mapped.append(7)
            else:
                print("ERROR: region not found")
        y_pred_val_mapped = []
        for i in y_pred_val:
            i_str = str(i)  # Convert to string
            if i_str in region_1:
                y_pred_val_mapped.append(0)
            elif i_str in region_2:
                y_pred_val_mapped.append(1)
            elif i_str in region_3:
                y_pred_val_mapped.append(2)
            elif i_str in region_4:
                y_pred_val_mapped.append(3)
            elif i_str in region_5:
                y_pred_val_mapped.append(4)
            elif i_str in region_6:
                y_pred_val_mapped.append(5)
            elif i_str in region_7:
                y_pred_val_mapped.append(6)
            elif i_str in region_8:
                y_pred_val_mapped.append(7)
            else:
                print("ERROR: region not found")
        
        # calculate macro scores
        scores = calc_macro_results(y_pred_val_mapped, y_true_val_mapped)
        
        # get the values of the items from the dictionary
        # accent_encoder.ind2lab = {}
        y_true = [accent_encoder.ind2lab[i] for i in y_true_val]
        y_pred = [accent_encoder.ind2lab[i] for i in y_pred_val]

        # retrieve a list of classes
        classes = [i[1] for i in accent_encoder.ind2lab.items()]
        classes_in_test_set = set(y_true)
        classes_in_mapped_test_set = set(y_true_val_mapped)
        
        with open(
            f"{hparams['output_folder']}/classification_report_{set_name}.txt", "w"
        ) as f:
            f.write(classification_report(y_true, y_pred))
        print("output_folder: ", hparams["output_folder"])

        # print the classification report only for the classes in the test set
        with open(
            f"{hparams['output_folder']}/classification_report_{set_name}_filtered.txt", "w"
        ) as f:
            f.write(classification_report(y_true, y_pred, labels=list(classes_in_test_set)))
        
        # print the classification report for the mapped classes
        with open(
            f"{hparams['output_folder']}/classification_report_{set_name}_states.txt", "w"
        ) as f:
            f.write(classification_report(y_true_val_mapped, y_pred_val_mapped, labels=list(classes_in_mapped_test_set)))
        
        # create the confusion matrix and plot it
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        disp.ax_.tick_params(axis="x", labelrotation=45, labelsize=4)
        disp.ax_.tick_params(axis="y", labelsize=4)
        disp.figure_.savefig(
            f"{hparams['output_folder']}/conf_mat_{set_name}.png", dpi=300, bbox_inches='tight'
        )
        
        cm_mapped = confusion_matrix(y_true_val_mapped, y_pred_val_mapped, labels=[0, 1, 2, 3, 4, 5, 6, 7])
        disp_mapped = ConfusionMatrixDisplay(confusion_matrix=cm_mapped, display_labels=[0, 1, 2, 3, 4, 5, 6, 7])
        disp_mapped.plot()
        disp_mapped.ax_.tick_params(axis="x", labelrotation=45, labelsize=4)
        disp_mapped.ax_.tick_params(axis="y", labelsize=4)
        disp_mapped.figure_.savefig(
            f"{hparams['output_folder']}/conf_mat_{set_name}_states.png", dpi=300, bbox_inches='tight'
        )
        
        return y_true, y_pred

    print("test set:")
    # Load the best checkpoint for evaluation of test set
    test_stats = accid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
    # print_confusion_matrix(accid_brain, set_name="test_mini2")
    print_confusion_matrix(accid_brain, set_name="test")

    print("dev set:")
    # Load the best checkpoint for evaluation of dev set
    test_stats = accid_brain.evaluate(
        test_set=datasets["dev"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
    print_confusion_matrix(accid_brain, set_name="dev")

    # ipdb.set_trace()
