"""
Functions for singing voice transcription

Authors
* Xiangming Gu 2022
"""
import numpy as np


def note2frame(gt_data, length, frame_size=1/49.8, pitch_shift=0):
    """
    This function transforms the note-level annotations into the frame-level annotations
    Adapted from https://github.com/york135/singing_transcription_ICASSP2021/blob/master/AST/data_utils/audio_dataset.py
    """
    new_label = []

    cur_note = 0
    cur_note_onset = gt_data[cur_note][0]
    cur_note_offset = gt_data[cur_note][1]
    cur_note_pitch = gt_data[cur_note][2] + pitch_shift

    # start from C2 (36) to B5 (83), total: 4 classes. This is a little confusing
    octave_start = 0
    octave_end = 3
    pitch_class_num = 12
    # frame_size = 1/ 49 # 1024.0 / 44100.0

    for i in range(length):
        cur_time = i * frame_size

        if abs(cur_time - cur_note_onset) <= (frame_size / 2.0):
            # First dim : onset
            # Second dim : no pitch
            if i == 0 or new_label[-1][0] != 1:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [1, 0, my_oct, my_pitch_class]
                new_label.append(label)
            else:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [0, 0, my_oct, my_pitch_class]
                new_label.append(label)

        elif cur_time < cur_note_onset or cur_note >= len(gt_data):
            # For the frame that doesn't belong to any note
            label = [0, 1, octave_end+1, pitch_class_num]
            new_label.append(label)

        elif abs(cur_time - cur_note_offset) <= (frame_size / 2.0):
            # For the offset frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num
            label = [0, 1, my_oct, my_pitch_class]

            cur_note = cur_note + 1
            if cur_note < len(gt_data):
                cur_note_onset = gt_data[cur_note][0]
                cur_note_offset = gt_data[cur_note][1]
                cur_note_pitch = gt_data[cur_note][2] + pitch_shift
                if abs(cur_time - cur_note_onset)  <= (frame_size / 2.0):
                    my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                    my_pitch_class = cur_note_pitch % pitch_class_num
                    label[0] = 1
                    label[1] = 0
                    label[2] = my_oct
                    label[3] = my_pitch_class

            new_label.append(label)

        else:
            # For the voiced frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num

            label = [0, 0, my_oct, my_pitch_class]
            new_label.append(label)

    return np.array(new_label)


def frame2note(frame_info, onset_thres, offset_thres, frame_size=1/49.8):
    """
    This function transforms the frame-level predictions into the note-level predictions.
    Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format.
    Adapted from https://github.com/york135/singing_transcription_ICASSP2021/blob/master/AST/predictor.py
    """

    result = []
    current_onset = None
    pitch_counter = []

    last_onset = 0.0
    onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

    local_max_size = 3
    current_time = 0.0

    onset_seq_length = len(onset_seq)

    for i in range(len(frame_info)):

        current_time = frame_size*i
        info = frame_info[i]

        backward_frames = i - local_max_size
        if backward_frames < 0:
            backward_frames = 0

        forward_frames = i + local_max_size + 1
        if forward_frames > onset_seq_length - 1:
            forward_frames = onset_seq_length - 1

        # local max and more than threshold
        if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

            if current_onset is None:
                current_onset = current_time
                last_onset = info[0] - onset_thres

            else:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

                current_onset = current_time
                last_onset = info[0] - onset_thres
                pitch_counter = []

        elif info[1] >= offset_thres:  # If is offset
            if current_onset is not None:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                current_onset = None

                pitch_counter = []

        # If current_onset exist, add count for the pitch
        if current_onset is not None:
            final_pitch = int(info[2]* 12 + info[3])
            if info[2] != 4 and info[3] != 12:
            # if final_pitch != 60:
                pitch_counter.append(final_pitch)

    if current_onset is not None:
        if len(pitch_counter) > 0:
            result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
        current_onset = None

    return result


def frame2note_finegrain(frame_info, onset_thres, offset_thres, octave_class_num=4, pitch_class_num=12, frame_size=1/49.8):
    """
    This function transforms the frame-level predictions into the note-level predictions.
    Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format.
    Adapted from https://github.com/york135/singing_transcription_ICASSP2021/blob/master/AST/predictor.py
    """

    result = []
    current_onset = None
    pitch_counter = []

    last_onset = 0.0
    onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

    local_max_size = 3
    current_time = 0.0

    onset_seq_length = len(onset_seq)

    for i in range(len(frame_info)):

        current_time = frame_size*i
        info = frame_info[i]

        backward_frames = i - local_max_size
        if backward_frames < 0:
            backward_frames = 0

        forward_frames = i + local_max_size + 1
        if forward_frames > onset_seq_length - 1:
            forward_frames = onset_seq_length - 1

        # local max and more than threshold
        if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

            if current_onset is None:
                current_onset = current_time
                last_onset = info[0] - onset_thres

            else:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

                current_onset = current_time
                last_onset = info[0] - onset_thres
                pitch_counter = []

        elif info[1] >= offset_thres:  # If is offset
            if current_onset is not None:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                current_onset = None

                pitch_counter = []

        # If current_onset exist, add count for the pitch
        if current_onset is not None:
            final_pitch = float(info[2]* 12 + info[3] * 12 / pitch_class_num)
            if info[2] != octave_class_num and info[3] != pitch_class_num:
            # if final_pitch != 60:
                pitch_counter.append(final_pitch)

    if current_onset is not None:
        if len(pitch_counter) > 0:
            result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
        current_onset = None

    return result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count