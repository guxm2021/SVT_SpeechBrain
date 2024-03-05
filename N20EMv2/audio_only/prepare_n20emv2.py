"""
Data preparation for datasets of automatic music transcription

Authors
* Xiangming Gu 2022
"""
import os
import csv
import json
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from utils import note2frame
from speechbrain.dataio.dataio import merge_csvs
SAMPLERATE = 16000


def prepare_frame_anno(folder, frame_rate=49.8):
    """
    This function processes the frame-level annotations for each song
    """
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        anno = annotations[entry]["midi"]
        json_path = os.path.join(folder_data, entry, "note_anno.json")
        # save json file
        with open(json_path, "w") as f:
            json.dump(anno, f)
        f.close()
        # save frame-level annotations
        wav_file = os.path.join(folder_data, entry, "vocals.wav")
        audio, fs = torchaudio.load(wav_file)
        assert fs == SAMPLERATE
        assert audio.shape[0] == 1
        duration = audio.shape[1] / SAMPLERATE
        length = round(duration * frame_rate)
        frame_label = note2frame(gt_data=anno, length=length, frame_size=1/frame_rate)
        # print(length)
        assert frame_label.shape[0] == length
        # save frame-level annotation
        os.makedirs(os.path.join(folder_data, entry, "audio_anno", str(frame_rate) + "fps"), exist_ok=True)
        npy_path = os.path.join(folder_data, entry, "audio_anno", str(frame_rate) + "fps", "audio_frame_anno.npy")
        np.save(npy_path, frame_label)


def prepare_csv_n20emv2(folder, csv_folder="./data", dur_thrd=5):
    """
    This function creates csv files for speechbrain to process, dur_thrd is the threshold for the duration
    """

    # initialize the csv lines
    csv_train_lines = [["ID", "duration", "wav", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_valid_lines = [["ID", "duration", "wav", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_test_lines = [["ID", "duration", "wav", "utter_id", "utter_num", "frame_anno", "song_anno"]]
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
        audio_path = os.path.join(folder_data, entry, "vocals.wav")
        anno_path = os.path.join(folder_data, entry, "frame_anno.npy")
        song_anno_path = os.path.join(folder_data, entry, "note_anno.json")

        # load the audio
        audio, fs = torchaudio.load(audio_path)  # audio: [1, N] for mono or [2, N] for stero
        assert fs == SAMPLERATE
        duration = audio.shape[1] / SAMPLERATE

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
                ID, str(dur), audio_path, str(i), str(utter_num), anno_path, song_anno_path,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, help="the threshold for duration")
    parser.add_argument("--frame_rate", type=float, default=49.8, help="The frame-rate for SSL models")
    parser.add_argument("--n20emv2", type=str, default="/path/to/N20EMv2", help="The path to save N20EMv2 dataset")
    args = parser.parse_args()
    
    prepare_frame_anno(folder=args.n20emv2, frame_rate=args.frame_rate)
    prepare_csv_n20emv2(folder=args.n20emv2, dur_thrd=args.duration)
    save_folder = os.path.join("./data", "dur_" + str(args.duration) + "s")
    
    merge_files = ["mir_st500_train.csv", "n20em_train.csv"]
    merge_name = "mix_train.csv"
    merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )