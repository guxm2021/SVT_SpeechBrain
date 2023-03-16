"""
Data preparation for N20EMv2 of singing voice transcription
The input to model needs to be spectrum features
Authors
* Xiangming Gu 2022
"""
import os
import csv
import json
import torch
import argparse
from tqdm import tqdm
SAMPLERATE=16000


def prepare_csv_n20emv2_feat(folder, csv_folder="./data_feat", dur_thrd=5):
    """
    This function creates csv files for speechbrain to process, dur_thrd is the threshold for the duration
    """

    # initialize the csv lines
    csv_train_lines = [["ID", "duration", "audio", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_valid_lines = [["ID", "duration", "audio", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_test_lines = [["ID", "duration", "audio", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    # load the annotations
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        split = annotations[entry]["split"]
        audio_path = os.path.join(folder_data, entry, "noise_data", "clean_feats.pt")
        video_path = os.path.join(folder_data, entry, "noise_data", "video_feats.pt")
        anno_path = os.path.join(folder_data, entry, "frame_anno.npy")
        song_anno_path = os.path.join(folder_data, entry, "note_anno.json")

        # load the audio
        audio = torch.load(audio_path)
        video = torch.load(video_path)
        frame1 = audio.shape[0]
        frame2 = video.shape[0]
        duration = frame1 / 49.8  # audio frame-rate 

        # split the whole song into utterances
        utter_num = round(duration / dur_thrd)
        for i in range(1, utter_num+1):
            ID = entry + "_" + str(i)
            if i == utter_num:
                dur = duration - (utter_num - 1) * dur_thrd
                assert 0 < dur <= dur_thrd * 3 / 2
            else:
                dur = dur_thrd
            csv_line = [
                ID, str(dur), audio_path, video_path, str(i), str(utter_num), anno_path, song_anno_path,
            ]
            if split == "train":
                csv_train_lines.append(csv_line)
            elif split == "valid":
                csv_valid_lines.append(csv_line)
            elif split == "test":
                csv_test_lines.append(csv_line)
    # save csv files
    save_folder = os.path.join(csv_folder, "dur_" + str(dur_thrd) + "s")
    os.makedirs(save_folder, exist_ok=True)
    save_train_path = os.path.join(save_folder, "n20em_train.csv")
    save_valid_path = os.path.join(save_folder, "n20em_valid.csv")
    save_test_path = os.path.join(save_folder, "n20em_test.csv")
    # train
    with open(save_train_path, mode="w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_train_lines:
            csv_writer.writerow(line)
    # valid
    with open(save_valid_path, mode="w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_valid_lines:
            csv_writer.writerow(line)
    # test
    with open(save_test_path, mode="w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_test_lines:
            csv_writer.writerow(line)

if __name__ == "__main__":
    # prepare_frame_anno(folder="/data1/guxm/svt_datasets/n20em/song_level")
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, help="the threshold for duration")
    parser.add_argument("--n20emv2", type=str, default="/path/to/N20EMv2", help="The path to save N20EMv2 dataset")
    args = parser.parse_args()
    prepare_csv_n20emv2_feat(folder=args.n20emv2, dur_thrd=args.duration)