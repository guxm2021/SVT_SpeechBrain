import os
import json
import glob
import torch
import random
import argparse
from scipy.io import wavfile
from tqdm import tqdm
import torchaudio
import numpy as np
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
)
SAMPLERATE = 16000


def resample_wav(folder):
    """
    This function is used to re-sample the raw audio into the 16kHz
    """
    # folder_notes = os.path.join(folder, "notes")
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    folder_accomp = os.path.join(folder, "accomp", "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        accomp_file = os.path.join(folder_accomp, entry, "accomp.wav")
        resample_accomp_file = os.path.join(folder_data, entry, "accomp.wav")
        audio, fs = torchaudio.load(accomp_file)
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
        torchaudio.save(resample_accomp_file, audio_resample, SAMPLERATE)


def align_audio_video(folder):
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    diffs = []
    for entry in tqdm(annotations.keys()):
        # extract features from the raw waveform
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        sample_rate, audio = wavfile.read(wav_path)
        assert sample_rate == SAMPLERATE and len(audio.shape) == 1
        duration1 = audio.shape[0] / SAMPLERATE
        video_path = os.path.join(folder_data, entry, "video_50fps.npy")
        video = np.load(video_path)
        duration2 = video.shape[0] / 50
        diff = abs(duration1 - duration2)
        diffs.append(diff)
    diffs = np.array(diffs)
    print(diffs.max())


def align_audio_accom(folder):
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    diffs = []
    for entry in tqdm(annotations.keys()):
        # extract features from the raw waveform
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        sample_rate, audio = wavfile.read(wav_path)
        assert sample_rate == SAMPLERATE and len(audio.shape) == 1
        duration1 = audio.shape[0] / SAMPLERATE
        accomp_path = os.path.join(folder_data, entry, "accomp.wav")
        sample_rate2, accomp = wavfile.read(accomp_path)
        assert sample_rate2 == SAMPLERATE and len(accomp.shape) == 1
        duration2 = accomp.shape[0] / SAMPLERATE
        diff = abs(duration1 - duration2)
        diffs.append(diff)
    diffs = np.array(diffs)
    print(diffs.max())


def synthesis_accomp(folder):
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        # create the folder for noise data
        os.makedirs(os.path.join(folder_data, entry, "noise_data", "accomp"), exist_ok=True)
        # load vocal and accompaniment
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        audio, sample_rate = torchaudio.load(wav_path)
        # print(audio.shape)
        assert sample_rate == SAMPLERATE and audio.shape[0] == 1
        length1 = audio.shape[1]
        accomp_path = os.path.join(folder_data, entry, "accomp.wav")
        accomp, sample_rate2 = torchaudio.load(accomp_path)
        assert sample_rate2 == SAMPLERATE and accomp.shape[0] == 1
        length2 = accomp.shape[1]
        assert length1 == length2  # the same length, no need to pad or truncate
        # synthesis the accompanied data
        for snr_db in [-10, -5, 0, 5, 10]:
            # Copy clean waveform to initialize noisy waveform
            sig = audio.clone()
            # Pick an SNR and use it to compute the mixture amplitude factors
            clean_amplitude = compute_amplitude(audio)
            noise_amplitude_factor = 1 / (dB_to_amplitude(snr_db) + 1)
            new_noise_amplitude = noise_amplitude_factor * clean_amplitude

            # Scale clean signal appropriately
            sig *= 1 - noise_amplitude_factor

            # Rescale and add
            noise_amplitude = compute_amplitude(accomp)
            accomp *= new_noise_amplitude / (noise_amplitude + 1e-14)
            sig += accomp

            # Save the noise data
            noise_data_path = os.path.join(folder_data, entry, "noise_data", "accomp", f"SNR_{snr_db}dB.wav")
            torchaudio.save(noise_data_path, sig, SAMPLERATE)
            # print(sig.shape)
        #     break
        # break


def synthesis_white(folder):
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        # create the folder for noise data
        os.makedirs(os.path.join(folder_data, entry, "noise_data", "white"), exist_ok=True)
        # load vocal and accompaniment
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        audio, sample_rate = torchaudio.load(wav_path)
        # print(audio.shape)
        assert sample_rate == SAMPLERATE and audio.shape[0] == 1
        white_noise = torch.randn_like(audio)
        
        # synthesis the accompanied data
        for snr_db in [-10, -5, 0, 5, 10]:
            # Copy clean waveform to initialize noisy waveform
            sig = audio.clone()
            # Pick an SNR and use it to compute the mixture amplitude factors
            clean_amplitude = compute_amplitude(audio)
            noise_amplitude_factor = 1 / (dB_to_amplitude(snr_db) + 1)
            new_noise_amplitude = noise_amplitude_factor * clean_amplitude

            # Scale clean signal appropriately
            sig *= 1 - noise_amplitude_factor

            # Rescale and add
            noise_amplitude = compute_amplitude(white_noise)
            white_noise *= new_noise_amplitude / (noise_amplitude + 1e-14)
            sig += white_noise

            # Save the noise data
            noise_data_path = os.path.join(folder_data, entry, "noise_data", "white", f"SNR_{snr_db}dB.wav")
            torchaudio.save(noise_data_path, sig, SAMPLERATE)
            # print(sig.shape)
        #     break
        # break


def synthesis_babble(folder, noise_folder, save_json_file="noise/babble.json", duration_thrd=10):
    # first select noise files
    noise_files = glob.glob(noise_folder + "/*/*wav")
    json_data = {}
    # step I: save json file for noise in train/valid/test split
    for file in tqdm(noise_files):
        noise, sample_rate = torchaudio.load(file)
        assert sample_rate == SAMPLERATE and noise.shape[0] == 1
        split = file.split("/")[-1].split("-")[0]
        duration = noise.shape[1] / SAMPLERATE
        if duration == 10:
            json_key = file.split("/")[-1]
            json_value = {
                "path": file,
                "split": split,
                "duration": duration
            }
            json_data[json_key] = json_value
    json_data_write = json.dumps(json_data, indent=2)
    with open(save_json_file, "w") as f:
        f.write(json_data_write)
    f.close()
    # 
    noise_pool_train = {}
    noise_pool_valid = {}
    noise_pool_test = {}
    for entry in json_data:
        path = json_data[entry]["path"]
        split = json_data[entry]["split"]
        noise, sample_rate = torchaudio.load(path)
        assert sample_rate == SAMPLERATE and noise.shape[0] == 1
        if split == "train":
            noise_pool_train[entry] = noise
        elif split == "valid":
            noise_pool_valid[entry] = noise
        elif split == "test":
            noise_pool_test[entry] = noise
    # step II: merge noise files
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        # create the folder for noise data
        os.makedirs(os.path.join(folder_data, entry, "noise_data", "babble"), exist_ok=True)
        # load vocal and accompaniment
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        audio, sample_rate = torchaudio.load(wav_path)
        # print(audio.shape)
        assert sample_rate == SAMPLERATE and audio.shape[0] == 1
        length = round(np.ceil(audio.shape[1] / SAMPLERATE / duration_thrd))
        # print(length)
        # print(audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd)
        babble_noise = []
        for utter in range(length):
            if utter < length - 1:  # 10s
                # randomly select noise
                if split == "train":
                    noise_data = random.choice(list(noise_pool_train.values()))
                elif split == "valid":
                    noise_data = random.choice(list(noise_pool_valid.values()))
                elif split == "test":
                    noise_data = random.choice(list(noise_pool_test.values()))
                # padding
                assert noise_data.shape[1] <= round(SAMPLERATE * duration_thrd)
                pad1 = (round(SAMPLERATE * duration_thrd) - noise_data.shape[1]) // 2
                pad2 = round(SAMPLERATE * duration_thrd) - noise_data.shape[1] - pad1
                noise_data = torch.cat([torch.zeros((1, pad1)), noise_data, torch.zeros((1, pad2))], dim=1)
                # 
                babble_noise.append(noise_data)
            else: # endding
                # randomly select noise
                if split == "train":
                    noise_data = random.choice(list(noise_pool_train.values()))
                elif split == "valid":
                    noise_data = random.choice(list(noise_pool_valid.values()))
                elif split == "test":
                    noise_data = random.choice(list(noise_pool_test.values()))
                # padding
                if noise_data.shape[1] <= (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd):
                    pad1 = (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd - noise_data.shape[1]) // 2
                    pad2 = audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd - noise_data.shape[1] - pad1
                    noise_data = torch.cat([torch.zeros((1, pad1)), noise_data, torch.zeros((1, pad2))], dim=1)
                # truncate
                elif noise_data.shape[1] > (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd):
                    truncate = audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd
                    noise_data = noise_data[:, :truncate]
                # 
                babble_noise.append(noise_data)
                # print(noise_data.shape)
        babble_noise = torch.cat(babble_noise, dim=1)
        # print(natural_noise.shape)
        # print(audio.shape)
        assert babble_noise.shape[1] == audio.shape[1]
        # Save the noise data
        noise_data_path = os.path.join(folder_data, entry, "noise_data", "babble", "noise.wav")
        torchaudio.save(noise_data_path, babble_noise, SAMPLERATE)

        # synthesis the accompanied data
        for snr_db in [-10, -5, 0, 5, 10]:
            # Copy clean waveform to initialize noisy waveform
            sig = audio.clone()
            # Pick an SNR and use it to compute the mixture amplitude factors
            clean_amplitude = compute_amplitude(audio)
            noise_amplitude_factor = 1 / (dB_to_amplitude(snr_db) + 1)
            new_noise_amplitude = noise_amplitude_factor * clean_amplitude

            # Scale clean signal appropriately
            sig *= 1 - noise_amplitude_factor

            # Rescale and add
            noise_amplitude = compute_amplitude(babble_noise)
            babble_noise *= new_noise_amplitude / (noise_amplitude + 1e-14)
            sig += babble_noise

            # Save the noise data
            noise_data_path = os.path.join(folder_data, entry, "noise_data", "babble", f"SNR_{snr_db}dB.wav")
            torchaudio.save(noise_data_path, sig, SAMPLERATE)
    


def synthesis_natural(folder, noise_folder, save_json_file="noise/natural.json", duration_thrd=10):
    # first select noise files
    noise_files1 = glob.glob(noise_folder + "/free-sound/*wav")
    noise_files2 = glob.glob(noise_folder + "/sound-bible/*wav")
    # Step I: save json file for noise in train/valid/test split
    json_data1 = {}
    json_data2 = {}
    for file in noise_files1:
        noise, sample_rate = torchaudio.load(file)
        assert sample_rate == SAMPLERATE and noise.shape[0] == 1
        duration = noise.shape[1] / SAMPLERATE
        key = file.split("/")[-2] + "/" + file.split("/")[-1]
        value = {
            "path": file,
            "split": None,
            "duration": duration,
        }
        if 1 <= duration <= 10:
            json_data1[key] = value
    for file in noise_files2:
        noise, sample_rate = torchaudio.load(file)
        assert sample_rate == SAMPLERATE and noise.shape[0] == 1
        duration = noise.shape[1] / SAMPLERATE
        key = file.split("/")[-2] + "/" + file.split("/")[-1]
        value = {
            "path": file,
            "split": None,
            "duration": duration,
        }
        if 1 <= duration <= 10:
            json_data2[key] = value
    entry_shuffle1 = list(json_data1.keys())
    entry_shuffle2 = list(json_data2.keys())
    random.shuffle(entry_shuffle1)
    random.shuffle(entry_shuffle2)
    iter = 0
    for entry in entry_shuffle1:
        if iter <= len(entry_shuffle1) * 3 / 4:
            json_data1[entry]["split"] = "train"
        elif len(entry_shuffle1) * 3 / 4 < iter <= len(entry_shuffle1) * 7 / 8:
            json_data1[entry]["split"] = "valid"
        else:
            json_data1[entry]["split"] = "test"
        iter += 1
    
    iter = 0
    for entry in entry_shuffle2:
        if iter <= len(entry_shuffle2) * 3 / 4:
            json_data2[entry]["split"] = "train"
        elif len(entry_shuffle2) * 3 / 4 < iter <= len(entry_shuffle2) * 7 / 8:
            json_data2[entry]["split"] = "valid"
        else:
            json_data2[entry]["split"] = "test"
        iter += 1
    json_data1.update(json_data2)
    json_data_write = json.dumps(json_data1, indent=2)
    with open(save_json_file, "w") as f:
        f.write(json_data_write)
    f.close()
    # 
    noise_pool_train = {}
    noise_pool_valid = {}
    noise_pool_test = {}
    for entry in json_data1:
        path = json_data1[entry]["path"]
        split = json_data1[entry]["split"]
        noise, sample_rate = torchaudio.load(path)
        assert sample_rate == SAMPLERATE and noise.shape[0] == 1
        if split == "train":
            noise_pool_train[entry] = noise
        elif split == "valid":
            noise_pool_valid[entry] = noise
        elif split == "test":
            noise_pool_test[entry] = noise

    # step II: merge noise data
    json_file = os.path.join(folder, "annotations.json")
    folder_data = os.path.join(folder, "data")
    # open ground truth data
    with open(json_file) as f:
        annotations = json.load(f)
    f.close()
    # traverse the whole dataset
    for entry in tqdm(annotations.keys()):
        # create the folder for noise data
        os.makedirs(os.path.join(folder_data, entry, "noise_data", "natural"), exist_ok=True)
        # load vocal and accompaniment
        wav_path = os.path.join(folder_data, entry, "vocals.wav")
        audio, sample_rate = torchaudio.load(wav_path)
        split = annotations[entry]["split"]
        # print(audio.shape)
        assert sample_rate == SAMPLERATE and audio.shape[0] == 1
        length = round(np.ceil(audio.shape[1] / SAMPLERATE / duration_thrd))
        # print(length)
        # print(audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd)
        natural_noise = []
        for utter in range(length):
            if utter < length - 1:  # 10s
                # randomly select noise
                if split == "train":
                    noise_data = random.choice(list(noise_pool_train.values()))
                elif split == "valid":
                    noise_data = random.choice(list(noise_pool_valid.values()))
                elif split == "test":
                    noise_data = random.choice(list(noise_pool_test.values()))
                # padding
                assert noise_data.shape[1] <= round(SAMPLERATE * duration_thrd)
                pad1 = (round(SAMPLERATE * duration_thrd) - noise_data.shape[1]) // 2
                pad2 = round(SAMPLERATE * duration_thrd) - noise_data.shape[1] - pad1
                noise_data = torch.cat([torch.zeros((1, pad1)), noise_data, torch.zeros((1, pad2))], dim=1)
                # 
                natural_noise.append(noise_data)
            else: # endding
                # randomly select noise
                if split == "train":
                    noise_data = random.choice(list(noise_pool_train.values()))
                elif split == "valid":
                    noise_data = random.choice(list(noise_pool_valid.values()))
                elif split == "test":
                    noise_data = random.choice(list(noise_pool_test.values()))
                # padding
                if noise_data.shape[1] <= (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd):
                    pad1 = (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd - noise_data.shape[1]) // 2
                    pad2 = audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd - noise_data.shape[1] - pad1
                    noise_data = torch.cat([torch.zeros((1, pad1)), noise_data, torch.zeros((1, pad2))], dim=1)
                # truncate
                elif noise_data.shape[1] > (audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd):
                    truncate = audio.shape[1] - (length - 1) * SAMPLERATE * duration_thrd
                    noise_data = noise_data[:, :truncate]
                # 
                natural_noise.append(noise_data)
                # print(noise_data.shape)
        natural_noise = torch.cat(natural_noise, dim=1)
        # print(natural_noise.shape)
        # print(audio.shape)
        assert natural_noise.shape[1] == audio.shape[1]
        # Save the noise data
        noise_data_path = os.path.join(folder_data, entry, "noise_data", "natural", "noise.wav")
        torchaudio.save(noise_data_path, natural_noise, SAMPLERATE)
        # synthesis the accompanied data
        for snr_db in [-10, -5, 0, 5, 10]:
            # Copy clean waveform to initialize noisy waveform
            sig = audio.clone()
            # Pick an SNR and use it to compute the mixture amplitude factors
            clean_amplitude = compute_amplitude(audio)
            noise_amplitude_factor = 1 / (dB_to_amplitude(snr_db) + 1)
            new_noise_amplitude = noise_amplitude_factor * clean_amplitude

            # Scale clean signal appropriately
            sig *= 1 - noise_amplitude_factor

            # Rescale and add
            noise_amplitude = compute_amplitude(natural_noise)
            natural_noise *= new_noise_amplitude / (noise_amplitude + 1e-14)
            sig += natural_noise

            # Save the noise data
            noise_data_path = os.path.join(folder_data, entry, "noise_data", "natural", f"SNR_{snr_db}dB.wav")
            torchaudio.save(noise_data_path, sig, SAMPLERATE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for noise simulation")
    parser.add_argument("--musan", type=str, default="/path/to/MUSAN", help="The path to save MUSAN dataset")
    parser.add_argument("--n20emv2", type=str, default="/path/to/N20EMv2", help="The path to save N20EMv2 dataset")
    args = parser.parse_args()
    random.seed(args.seed)
    synthesis_accomp(folder=args.n20emv2)
    synthesis_white(folder=args.n20emv2)
    synthesis_babble(folder=args.n20emv2, noise_folder=args.musan)
    synthesis_natural(folder=args.n20emv2, noise_folder=args.musan)
    