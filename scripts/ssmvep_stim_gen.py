import os
import time
import datetime
from datetime import timezone
import multiprocessing as mp

import numpy as np
import pandas as pd

from psychopy import visual, event, core

screen_size = [1750, 950]

# SSMVEP stimulus settings
rcycles = 14
M = 12
D = 10
L = 18
stimulus_size = 64
checkerboard_size = (1.25*128, 1.25*128)
xylim = 2 * np.pi * rcycles
x1, y1 = np.meshgrid(np.linspace(-xylim, xylim, stimulus_size), np.linspace(-xylim, xylim, stimulus_size))
angle_xy = np.arctan2(x1, y1)
temp_circle = (x1 ** 2 + y1 ** 2)
radius_values = np.sqrt(temp_circle)
circle1 = (temp_circle <= xylim ** 2) * 1
circle2 = (temp_circle >= 80) * 1
mask = circle1 * circle2
first_term = (np.pi * radius_values / D)
second_term = np.cos(angle_xy * M)

def get_frame_movement_phase(frame_number, stimulus_frequency, screen_refresh_rate):
    movement_phase = ((np.pi / 2) + (np.pi / 2) * np.sin((2 * np.pi * frame_number * (stimulus_frequency / (2 * screen_refresh_rate))) - (np.pi / 2)))
    checks = np.sign(np.cos(first_term + movement_phase*(L / D)) * second_term) * mask
    
    return checks

def generate_radial_stimulus_list(win, positions_list):
    stimulus_list = []
    for stim_position in positions_list:
        wedge = visual.GratingStim(win, size=checkerboard_size[0], pos=stim_position, units='pix')  
        stimulus_list.append(wedge)
        
    return stimulus_list

def get_utc_time():
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    
    return utc_timestamp

def save_timestamps_dataframe(event_times_df, base_folder_path, stimulus_type):
    utc_timestamp = get_utc_time()
    savefile_path = f'{base_folder_path}\{stimulus_type}_timestamps_event_id_{int(utc_timestamp)}.csv'
    event_times_df.to_csv(savefile_path, index=False)
    
def run_ssmvep_protocol(base_folder_path=None, stimulus_sequence=None,
                        positions_list=None, screen_refresh_rate=60, frequencies_list=None, 
                        stimulus_size=(64, 64), cue_period=2, stimulation_period=8, break_period=5):
    
    event_times_df = pd.DataFrame({'event_name': [], 'timestamp': [], 'utc_time': [], 'lsl_time': []})
    win = visual.Window(screen_size, color=(0.0, 0.0, 0.0))    
    stimulus_list = generate_radial_stimulus_list(win, positions_list)

    for sequence_idx, stimulus_id in enumerate(stimulus_sequence):
        event_times_df = event_times_df.append({'event_name': 'cue_start', 'utc_time': get_utc_time()}, ignore_index=True)
        message = visual.TextStim(win, text='1', height=0.1, color=(0, 1, 0))
        message.setAutoDraw(True)
        message.pos = (
            positions_list[stimulus_id - 1][0] / screen_size[0] * 2,
            positions_list[stimulus_id - 1][1] / screen_size[1] * 2)
        win.flip()
        time.sleep(cue_period)
        message.setAutoDraw(False)
        event_times_df = event_times_df.append({'event_name': f'stim_{stimulus_id}', 'utc_time': get_utc_time()}, ignore_index=True)
        for frame_number in range(1, int(stimulation_period * screen_refresh_rate)):
            if not event.getKeys():
                for stimulus_idx, stimulus in enumerate(stimulus_list):
                    stimulus.tex = get_frame_movement_phase(frame_number,
                                                            frequencies_list[stimulus_idx],
                                                            screen_refresh_rate)
                for stimulus_idx, stimulus in enumerate(stimulus_list):
                    stimulus.draw()
                win.flip()
            else:
                win.close()
                save_timestamps_dataframe(event_times_df, base_folder_path, 'ssmvep')
                core.quit()
        
        win.flip()

        event_times_df = event_times_df.append({'event_name': 'break_start', 'utc_time': get_utc_time()}, ignore_index=True)
        core.wait(break_period)
        event_times_df = event_times_df.append({'event_name': 'break_end', 'utc_time': get_utc_time(),}, ignore_index=True)
    save_timestamps_dataframe(event_times_df, base_folder_path, 'ssmvep')


            
class SSMVEPExpProtocol():
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
        print('self.stimulus_sequence: ', self.stimulus_sequence)
        proc = mp.Process(target=run_ssmvep_protocol, args=[self.basefolder_path, self.stimulus_sequence,
                                                            self.stimulus_positions_list, self.screen_refresh_rate, self.stimulus_frequency_list,
                                                            self.stimulus_size, self.cue_period, self.stimulation_period, self.break_period])
        proc.start()


if __name__ == '__main__':
    number_of_trials = 20
    stimulus_size = [64, 64]
    stimulus_frequencies = [10, 7.5]
    stimulus_positions = [(-320, 0), (320, 0)]
    cue_period = 2
    stimulation_period = 8
    break_period = 5
    screen_refresh_rate = 60
    
    ssmvep = SSMVEPExpProtocol(number_of_trials, stimulus_size, stimulus_frequencies, stimulus_positions,
                               cue_period, stimulation_period, break_period, screen_refresh_rate)
    ssmvep.run()