"""
Data preparation for datasets of singing voice transcription

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
SAMPLERATE = 16000


def source_separation_MIR_ST500(dataset_dir, spleeter_dir):
    """
    Sample code for source separation for MIR-ST500 dataset
    """
    from spleeter.separator import Separator
    import warnings
    separator = Separator('spleeter:2stems')

    # for the_dir in os.listdir(dataset_dir):
    for i in tqdm(range(1, 500+1)):
        mix_path = os.path.join(dataset_dir, f"Mixture{i}.m4a")
    
        y, sr = librosa.core.load(mix_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

        waveform = np.expand_dims(y, axis=1)

        prediction = separator.separate(waveform)
        voc = librosa.core.to_mono(prediction["vocals"].T)
        voc = np.clip(voc, -1.0, 1.0)

        acc = librosa.core.to_mono(prediction["accompaniment"].T)
        acc = np.clip(acc, -1.0, 1.0)

        import soundfile
        os.makedirs(os.path.join(spleeter_dir, str(i)), exist_ok=True)
        soundfile.write(os.path.join(spleeter_dir, str(i), "Vocal.wav"), voc, 44100, subtype='PCM_16')
        soundfile.write(os.path.join(spleeter_dir, str(i), "Inst.wav"), acc, 44100, subtype='PCM_16')

        
def resample_dataset(folder, save_folder):
    """
    Sample code for resampling the vocal audio data from 44.1kHz to 16kHz, which is the input requirement of wav2vec 2.0
    """
    os.makedirs(save_folder, exist_ok=True)
    for dir in tqdm(os.listdir(folder)):
        audio_path = os.path.join(folder, dir, "vocals.wav")
        os.makedirs(os.path.join(save_folder, dir), exist_ok=True)
        save_path = os.path.join(save_folder, dir, "vocals.wav")
        audio, fs = torchaudio.load(audio_path)  # audio: [1, N] for mono or [2, N] for stero
        # resample
        if fs != SAMPLERATE:
            # resample
            audio_resample = torchaudio.transforms.Resample(orig_freq=fs, new_freq=SAMPLERATE)(audio)
        else:
            audio_resample = audio
        
        # mono
        if audio_resample.shape[0] == 2:
            # stero input
            audio_resample = audio_resample.mean(dim=0, keepdim=True)
        
        # save the file
        torchaudio.save(save_path, audio_resample, SAMPLERATE)


def prepare_frame_anno(gt_file, folder, frame_rate=49.8):
    """
    This function processes the frame-level annotation for each song, frame_rate is 49.8 Hz, which is approximately the frequency of wav2vec 2.0
    """
    # open ground truth data
    with open(gt_file) as json_data:
        gt = json.load(json_data)
    
    for dir in tqdm(os.listdir(folder)):
        anno = gt[dir]
        json_path = os.path.join(folder, dir, "annotation.json")
        # save json file
        with open(json_path, "w") as json_data:
            json.dump(anno, json_data)
        # save frame-level annotations
        wav_file = os.path.join(folder, dir, "vocals.wav")
        audio, fs = torchaudio.load(wav_file)
        assert fs == SAMPLERATE
        assert audio.shape[0] == 1
        duration = audio.shape[1] / SAMPLERATE
        length = round(duration * frame_rate)
        frame_label = note2frame(gt_data=anno, length=length, frame_size=1/frame_rate)
        # print(length)
        assert frame_label.shape[0] == length
        # save frame-level annotation
        npy_path = os.path.join(folder, dir, "frame_anno.npy")
        np.save(npy_path, frame_label)
        

def prepare_csv_benchmarks(folder, save_path, dur_thrd=5, gender_file=None):
    """
    This function creates the csv file of dataset for speechbrain with duration threshold
    """
    csv_lines = [["ID", "duration", "wav", "utter_id", "utter_num", "frame_anno", "song_anno", "sex"]]
    if gender_file is not None:
        with open(gender_file) as f:
            gender_info = json.load(f)
    for dir in tqdm(os.listdir(folder)):
        audio_path = os.path.join(folder, dir, "vocals.wav")
        anno_path = os.path.join(folder, dir, "frame_anno.npy")
        song_anno_path = os.path.join(folder, dir, "annotation.json")
        # anno_np = np.load(anno_path)
        # print(anno_np.shape)
        audio, fs = torchaudio.load(audio_path)  # audio: [1, N] for mono or [2, N] for stero
        assert fs == SAMPLERATE
        duration = audio.shape[1] / SAMPLERATE
        # determine the number of utterance
        utter_num = round(duration / dur_thrd)
        for i in range(1, utter_num+1):
            ID = dir + "_" + str(i)
            gender = gender_info[dir] if gender_file is not None else ""
            if i == utter_num:
                dur = duration - (utter_num - 1) * dur_thrd
                assert 0 < dur <= dur_thrd * 3 / 2
            else:
                dur = dur_thrd
            csv_line = [
                ID, str(dur), audio_path, str(i), str(utter_num), anno_path, song_anno_path, gender
            ]
            csv_lines.append(csv_line)
    
    with open(save_path, mode="w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_lines:
            csv_writer.writerow(line)


def prepare_all_SVT_datasets(args, save_folder="./data"):
    dur_thrd = args.duration
    # create folder
    csv_folder = os.path.join(save_folder, "dur_" + str(dur_thrd) + "s")
    os.makedirs(csv_folder, exist_ok=True)
    print(f"save to {csv_folder}")
    
    # Step I: prepare frame-level annotations

    # prepare MIR-ST500
    prepare_frame_anno(gt_file=os.path.join(args.mir_st500, "Annotations.json"), folder=os.path.join(args.mir_st500, "wav16kHz", "train"), frame_rate=args.frame_rate)
    prepare_frame_anno(gt_file=os.path.join(args.mir_st500, "Annotations.json"), folder=os.path.join(args.mir_st500, "wav16kHz", "test"), frame_rate=args.frame_rate)
    
    # prepare ISMIR2014 dataset
    prepare_frame_anno(gt_file=os.path.join(args.ismir, "Annotations.json"), folder=os.path.join(args.ismir, "wav16kHz"), frame_rate=args.frame_rate)

    # prepare TONAS dataset
    prepare_frame_anno(gt_file=os.path.join(args.tonas, "Annotations.json"), folder=os.path.join(args.tonas, "wav16kHz"), frame_rate=args.frame_rate)

    # Step II: prepare csv files

    # prepare MIR-ST500
    prepare_csv_benchmarks(folder=os.path.join(args.mir_st500, "wav16kHz", "train"), save_path=os.path.join(csv_folder, "mir_st500_train.csv"), dur_thrd=dur_thrd, gender_file=args.mir_st500_gender_file)
    prepare_csv_benchmarks(folder=os.path.join(args.mir_st500, "wav16kHz", "test"), save_path=os.path.join(csv_folder, "mir_st500_test.csv"), dur_thrd=dur_thrd, gender_file=args.mir_st500_gender_file)

    # prepare ISMIR2014 dataset
    prepare_csv_benchmarks(folder=os.path.join(args.ismir, "wav16kHz"), save_path=os.path.join(csv_folder, "ismir2014.csv"), dur_thrd=dur_thrd, gender_file=args.ismir_gender_file)

    # prepare TONAS dataset
    prepare_csv_benchmarks(folder=os.path.join(args.tonas, "wav16kHz"), save_path=os.path.join(csv_folder, "tonas.csv"), dur_thrd=dur_thrd)


if __name__ == "__main__":
    # prepare all SVT datasets
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, help="The threshold to split the songs")
    parser.add_argument("--frame_rate", type=float, default=49.8, help="The frame-rate for SSL models")
    parser.add_argument("--mir_st500", type=str, default="/path/to/MIR_ST500", help="The path to save MIR-ST500 dataset")
    parser.add_argument("--mir_st500_gender_file", type=str, default="/path/to/MIR_ST500/gender.json", help="The path to gender infomation for MIR-ST500 dataset")
    parser.add_argument("--ismir", type=str, default="/path/to/ISMIR2014", help="The path to save ISMIR2014 dataset")
    parser.add_argument("--ismir_gender_file", type=str, default="/path/to/ISMIR2014/gender.json", help="The path to gender infomation for ISMIR2014 dataset")
    parser.add_argument("--tonas", type=str, default="/path/to/TONAS", help="The path to save TONAS dataset")
    args = parser.parse_args()

    prepare_all_SVT_datasets(args)