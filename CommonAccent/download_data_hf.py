
"""
Script to download the CommonVoice dataset using Hugging Face

At Hugging Face: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
Download the dataset: https://commonvoice.mozilla.org/en/datasets

Author
------
 * Juan Pablo Zuluaga 2023
"""
import os
import argparse
import csv
from tqdm import tqdm

from datasets import load_dataset, load_from_disk

import warnings
warnings.filterwarnings("ignore")

# _COMMON_VOICE_FOLDER = "common_voice_11_0/common_voice_11_0.py"
# _COMMON_VOICE_FOLDER = "mozilla-foundation/common_voice_11_0"
# load from local copy:
_COMMON_VOICE_FOLDER = "/data/vokquant/data/common_voice_de/"


def prepare_cv_from_hf(output_folder, language="de"):
    """ function to prepare the datasets in <output-folder> """
    
    output_folder = os.path.join(output_folder, language)
    # create the output folder: in case is not present
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare the the common voice dataset in streaming mode
    # common_voice_ds = load_dataset(_COMMON_VOICE_FOLDER, language, streaming=True)
    # common_voice_ds = load_dataset("mozilla_common_voice", language, version="11.0", split="train+validation+test", streaming=True)
    common_voice_version = "mozilla-foundation/common_voice_11_0"
    language = "de"
    # streaming:
    # common_voice_ds = load_dataset(common_voice_version, language, streaming=True, splits=["validation"])
    # locally:
    common_voice_ds = load_dataset(common_voice_version, language, streaming=False)
    print("successfully loaded the dataset")
    
    # just select relevant splits: train/validation/test set
    splits = ["train", "validation", "test"]
    common_voice = {}
    
    # load, prepare and filter each split in streaming mode:
    for split in splits:
        # filter out samples without accent
        ds = common_voice_ds[split].filter( lambda x: x['accent'] != '')
        common_voice[split] = ds
    
    print("successfully filtered the dataset")
    for dataset in common_voice:
        csv_lines = []
        # Starting index
        idx = 0
        for sample in tqdm(common_voice[dataset]):
            # get path and utt_id
            mp3_path = sample['path']
            utt_id = mp3_path.split(".")[-2].split("/")[-1]            
            
            # Create a row with metadata + transcripts
            csv_line = [
                idx,  # ID
                utt_id,  # Utterance ID
                mp3_path,  # File name
                sample["locale"],
                sample["accent"],
                sample["age"],
                sample["gender"],
                sample["sentence"], # transcript
            ]
            
            # Adding this line to the csv_lines list
            csv_lines.append(csv_line)
            # Increment index
            idx += 1
            # print("idx: ", idx)
        print(f"Split: {dataset}, Number of samples: {len(csv_lines)}")
        # CSV column titles
        csv_header = ["idx", "utt_id", "mp3_path", "language", "accent", "age", "gender", "transcript"]
        # Add titles to the list at indexx 0
        csv_lines.insert(0, csv_header)
        
        # Writing the csv lines
        csv_file = os.path.join(output_folder, dataset+'.tsv')
        print(f"Writing {csv_file}")
        with open(csv_file, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_lines:
                csv_writer.writerow(line)
    print(f"Prepare CommonVoice: for {language} in {output_folder}")

def main():
    # read input from CLI, you need to run it from the command lind
    parser = argparse.ArgumentParser()
    
    # reporting vars
    parser.add_argument(
        "--language",
        type=str,
        default='de',
        help="Language to load",
    )
    parser.add_argument(
        "output_folder",
        help="path of the output folder to store the csv files for each split",
    )
    args = parser.parse_args()
    
    # call the main function
    # prepare_cv_from_hf(output_folder=args.output_folder, language=args.language)
    

if __name__ == "__main__":
    # main()
    prepare_cv_from_hf(output_folder="/data/vokquant/data/common_voice_de/", language="de")