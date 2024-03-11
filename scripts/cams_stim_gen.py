import os
import math
import datetime
from datetime import timezone
import multiprocessing as mp

import numpy as np
import pandas as pd

from psychopy import visual, event, core

screen_size = [1750, 950]
ao_images_folder_path = '../images'

def get_ao_stimuli_paths(ao_images_folder_path):
    ao_stimuli_image_paths = []
    for image_idx in range(1, 17):
        ao_stimuli_image_paths.append(f'{ao_images_folder_path}/{image_idx}.jpg')
        
    return ao_stimuli_image_paths

def get_utc_time():
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    
    return utc_timestamp

def save_timestamps_dataframe(event_times_df, base_folder_path, stimulus_type):
    utc_timestamp = get_utc_time()
    savefile_path = f'{base_folder_path}\{stimulus_type}_timestamps_event_id_{int(utc_timestamp)}.csv'
    event_times_df.to_csv(savefile_path, index=False)


def run_cams_protocol(base_folder_path=None, stimulus_sequence=None,
                      positions_list=None, screen_refresh_rate=60, frequencies_list=None, 
                      stimulus_size=(128, 128), cue_period=2, stimulation_period=6,
                      break_period=4, ao_stimuli_image_paths=None):
    
    event_times_df = pd.DataFrame({'event_name': [], 'utc_time': []})

    win = visual.Window(screen_size, color=(0.0, 0.0, 0.0), multiSample=True)
    ao_stimulus_1 = []
    ao_stimulus_2 = []
    for img_idx in range(len(ao_stimuli_image_paths)):
        ao_stimulus_1.append(visual.ImageStim(win, image=ao_stimuli_image_paths[img_idx], size=stimulus_size, pos=positions_list[0]))
        ao_stimulus_2.append(visual.ImageStim(win, image=ao_stimuli_image_paths[img_idx], size=stimulus_size, pos=positions_list[1]))
    for sequence_idx, stimulus_id in enumerate(stimulus_sequence):
        img_idx_1 = 0
        img_idx_2 = 0

        event_times_df = event_times_df.append({'event_name': 'cue_start', 'utc_time': get_utc_time()}, ignore_index=True)
        message = visual.TextStim(win, text=f'{stimulus_id}', height=0.1, color=(0, 1, 0))
        message.setAutoDraw(True)
        message.pos = (positions_list[stimulus_id-1][0], positions_list[stimulus_id-1][1])
        win.flip()
        core.wait(cue_period)
        message.setAutoDraw(False)
        event_times_df = event_times_df.append({'event_name': f'stim_{stimulus_id}', 'utc_time': get_utc_time()}, ignore_index=True)
        
        for frame_number in range(0, int(stimulation_period * screen_refresh_rate)):
            if not event.getKeys():
                img_idx_1 = int((math.floor(frame_number/frequencies_list[0])) % len(ao_stimuli_image_paths))
                img_idx_2 = int((math.floor(frame_number/frequencies_list[1])) % len(ao_stimuli_image_paths))
                ao_stimulus_1[img_idx_1].draw()
                ao_stimulus_2[img_idx_2].draw()
                win.flip()
            else:
                win.close()
                save_timestamps_dataframe(event_times_df, base_folder_path, 'cams')
                core.quit()
        
        win.flip()

        event_times_df = event_times_df.append({'event_name': 'break_start', 'utc_time': get_utc_time()}, ignore_index=True)
        core.wait(break_period)
        event_times_df = event_times_df.append({'event_name': 'break_end', 'utc_time': get_utc_time(),}, ignore_index=True)
    save_timestamps_dataframe(event_times_df, base_folder_path, 'cams')
            

class CAMSExpProtocol():

    def __init__(self, number_of_trials, stimulus_size, stimulus_frequencies, stimulus_positions,
                 cue_period, stimulation_period, break_period, screen_refresh_rate):
        
        self.number_of_trials = number_of_trials
        self.stimulus_size = stimulus_size
        self.stimulus_frequency_list = stimulus_frequencies
        self.stimulus_positions_list = stimulus_positions
        self.cue_period = cue_period
        self.stimulation_period = stimulation_period
        self.break_period = break_period
        self.screen_refresh_rate = screen_refresh_rate
        self.basefolder_path = os.getcwd()
        
        number_of_stimuli = len(self.stimulus_frequency_list)
        self.stimulus_sequence = np.tile(np.arange(1, number_of_stimuli + 1), self.number_of_trials)
        np.random.shuffle(self.stimulus_sequence)
    
    def run(self):
        print('Cue: ', self.cue_period, 'Stim: ', self.stimulation_period, 'Break: ', self.break_period)
        self.ao_stimuli_image_paths = get_ao_stimuli_paths(ao_images_folder_path)
        print('self.stimulus_sequence: ', self.stimulus_sequence)
        proc = mp.Process(target=run_cams_protocol, args=[self.basefolder_path, self.stimulus_sequence,
                                                          self.stimulus_positions_list, self.screen_refresh_rate, self.stimulus_frequency_list,
                                                          self.stimulus_size, self.cue_period, self.stimulation_period, self.break_period,
                                                          self.ao_stimuli_image_paths])
        proc.start()


if __name__ == '__main__':
    number_of_trials = 20
    stimulus_size = [0.5, 0.5]
    stimulus_frequencies = [5, 7]
    stimulus_positions = [(-0.5, 0), (0.5, 0)]
    cue_period = 2
    stimulation_period = 8
    break_period = 5
    screen_refresh_rate = 60
    cams = CAMSExpProtocol(number_of_trials, stimulus_size, stimulus_frequencies, stimulus_positions,
                           cue_period, stimulation_period, break_period, screen_refresh_rate)
    cams.run()