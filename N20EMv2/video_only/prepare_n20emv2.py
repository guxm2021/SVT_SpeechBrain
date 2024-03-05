"""
Data preparation for datasets of automatic music transcription

Authors
* Xiangming Gu 2022
"""
import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils import note2frame


def prepare_frame_anno(folder, frame_rate=50):
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
        # load video file
        video_file = os.path.join(folder_data, entry, "video_" + str(frame_rate) + "fps.npy")
        video = np.load(video_file)
        # compute duration and length
        length = video.shape[0]
        frame_label = note2frame(gt_data=anno, length=length, frame_size=1/frame_rate)
        assert frame_label.shape[0] == length
        # save frame-level annotation
        os.makedirs(os.path.join(folder_data, entry, "video_anno", str(frame_rate) + "fps"), exist_ok=True)
        frame_anno_path = os.path.join(folder_data, entry, "video_anno", str(frame_rate) + "fps", "video_frame_anno.npy")
        np.save(frame_anno_path, frame_label)


def prepare_csv_n20emv2(folder, csv_folder="./data", frame_rate=50, dur_thrd=5):
    """
    This function creates csv files for speechbrain to process, dur_thrd is the threshold for the duration
    """

    # initialize the csv lines
    csv_train_lines = [["ID", "duration", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_valid_lines = [["ID", "duration", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
    csv_test_lines = [["ID", "duration", "video", "utter_id", "utter_num", "frame_anno", "song_anno"]]
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
        video_path = os.path.join(folder_data, entry, "video_" + str(frame_rate) + "fps.npy")
        anno_path = os.path.join(folder_data, entry, "video_anno", str(frame_rate) + "fps", "video_frame_anno.npy")
        song_anno_path = os.path.join(folder_data, entry, "note_anno.json")

        # load the video
        video = np.load(video_path)
        duration = video.shape[0] / frame_rate

        # split the whole song into utterances
        is_end = False
        cur_i = 1
        cur_time = 0
        utter_lines = []
        stride = dur_thrd
        while not is_end:
            ID = entry + "_" + str(cur_i)
            # whether is the end
            if duration - cur_time <= dur_thrd * 3 / 2:
                is_end = True
                dur = duration - cur_time
                utter_num = cur_i
            else:
                dur = dur_thrd
            
            # determine the csv_line
            utter_lines.append((ID, dur))
            
            # update variables
            cur_i = cur_i + 1
            cur_time = cur_time + stride
        
        for i in range(1, utter_num + 1):
            ID, dur = utter_lines[i - 1]
            csv_line = [
                ID, str(dur), video_path, str(i), str(utter_num), anno_path, song_anno_path,
            ]
            if split == "train":
                csv_train_lines.append(csv_line)
            elif split == "valid":
                csv_valid_lines.append(csv_line)
            elif split == "test":
                csv_test_lines.append(csv_line)

    # save csv files
    save_folder = os.path.join(csv_folder, "frame_rate" + str(frame_rate), "dur_" + str(dur_thrd) + "s")
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
    parser.add_argument("--frame_rate", type=int, default=50, help="the frame rate for log fbanks features")
    parser.add_argument("--duration", type=int, default=5, help="the threshold for duration")
    parser.add_argument("--n20emv2", type=str, default="/path/to/N20EMv2", help="The path to save N20EMv2 dataset")
    args = parser.parse_args()
    prepare_frame_anno(folder=args.n20emv2, frame_rate=args.frame_rate)
    prepare_csv_n20emv2(folder=args.n20emv2, frame_rate=args.frame_rate, dur_thrd=args.duration)