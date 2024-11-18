import os

asccd_data_path = os.path.join(os.path.dirname(__file__), 'data', 'asccd')
subject_data_path = os.path.join(os.path.dirname(__file__), 'subjects')
preanalysis_results_path = os.path.join(os.path.dirname(__file__), 'preanalysis_results')
asccd_data_path = os.path.join(root, 'data', 'asccd')
subject_data_path = os.path.join(root, 'subjects')
preanalysis_results_path = os.path.join(root, 'preanalysis_results')

import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from scipy.signal import resample

from cl_iotools import HTK
from . import util
from . import match_filter
from . import asccd
from . import bursc
from . import timit


def get_subject_block_path(subject, block):
    subject_block = subject + "_B" + str(block)
    return os.path.join(subject_data_path, subject, subject_block)


@util.save(preanalysis_results_path, "hg", "hg")
def preanalyze_data_HS(subject, block, hz, chans=np.arange(128), car=False):
    """Load high-gamma htks and downsample. Takes parameters subject, block, and hz
    """
    subject_block = subject + "_B" + str(block) + "_70_150.mat"
    print("Reading .mat")
    #hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band_noCAR"), chans)['data']
    hg_htk = sio.loadmat(os.path.join(subject_data_path, subject, subject_block))['bands']
    assert hz in [100, 25]
    resampling_factor = 400/hz
    print("Resampling")
    n = hg_htk.shape[1]
    y = np.floor(np.log2(n))
    nextpow2 = np.power(2, y+1)
    hg_htk = np.pad(hg_htk , ((0,0), (0, int(nextpow2-n))), mode='constant')
    hg = resample(hg_htk, np.int(hg_htk.shape[1]/resampling_factor), axis=1)
    hg = hg[:, :np.int(n/resampling_factor)]

    hg = nan_zscore_hg(hg)

    if car is True:
        car = np.expand_dims(np.nanmean(hg, axis=0), axis=0)
        hg_car = hg - np.repeat(car, 256, axis=0)
        hg_car = nan_zscore_hg(hg_car)
        hg = hg_car

    return hg


@util.save(preanalysis_results_path, "hg", "hg")
def preanalyze_data(subject, block, hz, chans=np.arange(128), car=False):
    """Load high-gamma htks and downsample. Takes parameters subject, block, and hz
    """
    subject_block = subject + "_B" + str(block)
    print("Reading HTKs")
    #hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band_noCAR"), chans)['data']
    hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band"), chans)[
        'data']
    assert hz in [100, 25]
    resampling_factor = 400/hz
    print("Resampling")
    n = hg_htk.shape[1]
    y = np.floor(np.log2(n))
    nextpow2 = np.power(2, y+1)
    hg_htk = np.pad(hg_htk , ((0,0), (0, int(nextpow2-n))), mode='constant')
    hg = resample(hg_htk, np.int(hg_htk.shape[1]/resampling_factor), axis=1)
    hg = hg[:, :np.int(n/resampling_factor)]

    hg = nan_zscore_hg(hg)

    if car is True:
        car = np.expand_dims(np.nanmean(hg, axis=0), axis=0)
        hg_car = hg - np.repeat(car, 256, axis=0)
        hg_car = nan_zscore_hg(hg_car)
        hg = hg_car

    return hg

def load_hg(subject, block, hz=100):
    subject_block = subject + "_B" + str(block)
    hg_path = os.path.join(preanalysis_results_path, subject_block + "_" + str(hz) + "hz_hg.mat")
    return sio.loadmat(hg_path)['hg']

def nan_zscore_hg(hg):
    hg_to_return = []
    for hg_chan in hg:
        z = np.copy(hg_chan)
        z[~np.isnan(z)] = zscore(z[~np.isnan(z)])
        hg_to_return.append(z)
    return np.array(hg_to_return)    

def get_bad_channels(subject, block):
    subject_block = subject + "_B" + str(block)
    bcs_path = os.path.join(subject_data_path, subject, subject_block, 'Artifacts', 'badChannels.txt')
    with open(bcs_path) as f:
        bad_channels = f.read().strip().split()
    return [int(b) for b in bad_channels]

def get_anin_HS(subject, block, anin_chan=2):
    subject_block = subject + "_B" + str(block) + "_anin.hdf5"
    anin_path = os.path.join(subject_data_path, subject, subject_block)
    anin_htk = h5py.File(anin_path, 'r')
    anin_signal = np.expand_dims(anin_htk['anin_data'][anin_chan-1], axis=0)
    anin_fs = int(np.array(anin_htk['anin_fs']))
    return anin_signal, anin_fs

def get_anin(subject, block, anin_chan=2):
    subject_block = subject + "_B" + str(block)
    anin_path = os.path.join(subject_data_path, subject, subject_block, "Analog", "ANIN" + str(anin_chan) + ".htk")  # 音频文件，ANIN2和3分别表示左右声道
    anin_htk = HTK.readHTK(anin_path)
    anin_signal = anin_htk['data']
    anin_fs = anin_htk['sampling_rate']
    return anin_signal, anin_fs

def load_times(subject_number, block):
    subject_block = util.get_subject_block(subject_number, block)
    times_path = os.path.join(preanalysis_results_path, subject_block + "_times.mat")
    return sio.loadmat(times_path)['times']

@util.save(preanalysis_results_path, "times", "times")
def get_times_HS(subject, block, asccd_block, anin_chan=2):
    anin_signal, anin_fs = get_anin_HS(subject, block, anin_chan)
    times = find_times(anin_signal, anin_fs, asccd_block)
    return times

@util.save(preanalysis_results_path, "times", "times")
def get_times(subject, block, asccd_block, anin_chan=2, wavpaths=None, nreps=1):
    anin_signal, anin_fs = get_anin(subject, block, anin_chan)
    times = find_times(anin_signal, anin_fs, asccd_block, wavpaths, nreps)
    return times

def find_times(anin_signal, anin_fs, asccd_block=1, wavpaths=None, nreps=1):
    if wavpaths is None:
        wavpaths = get_wavpaths_for_asccd_block(asccd_block)

    times = []
    for wavpath in wavpaths:
        evnts = match_filter.find_time_for_one(wavpath, anin_signal, anin_fs, nreps)
        times.append(evnts)

    times = np.concatenate(times)
    times = np.sort(times[:,0])[np.newaxis, :]
    return times

def get_blocks_for_task(subject, task="chinese"):
    if subject in ["HS8", "HS9", "HS10", "HS11", "HS2", "HS14"]:
        blocks = list(range(1, 7))
    elif subject in ["HS4"]:
        blocks = list(range(1, 5))
    elif subject in ["HS6"]:
        blocks = list(range(1, 6))
    elif subject in ["HS3"]:
        blocks = [1, 2, 3, 5, 6, 7]
    elif subject == "EC182":
        blocks = [8, 9, 12, 13, 16, 17]
    elif subject == "EC183":
        blocks = [130, 131, 135, 136, 137, 138]
    elif subject == "EC186":
        blocks = [30, 31, 34, 35, 36, 37]
    elif subject == "EC196":
        blocks = [7, 12, 13, 17, 23]
    elif subject == "LA14":
        blocks = [3]
    elif subject in ["HS24", "HS30", "HS31", "HS33"]:
        blocks = [1, 2, 3, 4]
    elif subject == "HS32":
        blocks = [5, 6, 7]
    return blocks

def load_all_hgs_times_names(subject, task='chinese'):
    if task == 'chinese':
        if subject in ["HS8", "HS9", "HS10", "HS11", "HS2", "HS14"]:
            all_hgs = [load_hg(subject, block) for block in range(1, 7)]
            all_times = [load_times(subject, block) for block in range(1, 7)]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 7)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["HS4"]:
            all_hgs = [load_hg(subject, block) for block in range(1, 5)]
            all_times = [load_times(subject, block) for block in range(1, 5)]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 5)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["HS6"]:
            all_hgs = [load_hg(subject, block) for block in range(1, 6)]
            all_times = [load_times(subject, block) for block in range(1, 6)]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 6)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["HS3"]:
            all_hgs = [load_hg(subject, block) for block in [1, 2, 3, 5, 6, 7]]
            all_times = [load_times(subject, block) for block in [1, 2, 3, 5, 6, 7]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 7)]
            return all_hgs, all_times, all_asccd_names
        elif subject == "EC182":
            all_hgs = [load_hg(subject, block) for block in [8, 9, 12, 13, 16, 17]]
            all_times = [load_times(subject, block) for block in [8, 9, 12, 13, 16, 17]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 7)]            
            return all_hgs, all_times, all_asccd_names
        elif subject == "EC183":
            all_hgs = [load_hg(subject, block) for block in [130, 131, 135, 136, 137, 138]]
            all_times = [load_times(subject, block) for block in [130, 131, 135, 136, 137, 138]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 7)]            
            return all_hgs, all_times, all_asccd_names
        elif subject == "EC186":
            all_hgs = [load_hg(subject, block) for block in [30, 31, 34, 35, 36, 37]]
            all_times = [load_times(subject, block) for block in [30, 31, 34, 35, 36, 37]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 7)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["EC196"]:
            all_hgs = [load_hg(subject, block) for block in [7, 12, 13, 17, 23]]
            all_times = [load_times(subject, block) for block in [7, 12, 13, 17, 23]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 6)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["LA14"]:
            all_hgs = [load_hg(subject, block) for block in [3]]
            all_times = [load_times(subject, block) for block in [3]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 2)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["HS24", "HS30", "HS31", "HS33"]:
            all_hgs = [load_hg(subject, block) for block in range(1, 5)]
            all_times = [load_times(subject, block) for block in range(1, 5)]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in range(1, 5)]
            return all_hgs, all_times, all_asccd_names
        elif subject in ["HS32"]:
            all_hgs = [load_hg(subject, block) for block in [5, 6, 7]]
            all_times = [load_times(subject, block) for block in [5, 6, 7]]
            all_asccd_names = [get_asccd_names_for_asccd_block(block) for block in [1, 2, 3]]
            return all_hgs, all_times, all_asccd_names
        else:
            print("not implemented for subject " + subject)
            return None
    elif task == 'english':
        if subject in ["HS8", "HS9", "HS10", "HS11"]:
            all_hgs = [load_hg(subject, block) for block in range(8, 11)]
            all_times = [load_times(subject, block) for block in range(8, 11)]
            all_names = [bursc.get_bursc_names_for_bursc_block(1), timit.get_timit_names_for_timit_block(1), timit.get_timit_names_for_timit_block(5)]
            all_hgs = [load_hg(subject, block) for block in range(9, 11)]
            all_times = [load_times(subject, block) for block in range(9, 11)]
            all_names = [timit.get_timit_names_for_timit_block(1), timit.get_timit_names_for_timit_block(5)]
            return all_hgs, all_times, all_names
        elif subject == "EC182":
            blocks = [5, 7, 11, 19, 15]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            bursc_blocks = [2, 3, 4, 5, 6]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC183":
            blocks = [78, 79, 67, 107, 49]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            bursc_blocks = [2, 3, 4, 5, 6]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC186":
            blocks = [2, 4, 16, 22, 15]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            bursc_blocks = [2, 3, 5, 6, 4]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC196":
            blocks = [1, 25, 2, 9, 5]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            bursc_blocks = [2, 3, 5, 6, 4]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject in ["LA14"]:
            blocks = [4]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = ['intraop1']
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC157":
            blocks = [1,2,3,4,5]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            bursc_blocks = [1,2,3,4,5]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC143":
            blocks = [1,2,4,6,7]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC193":
            blocks = [5,6,11,12,13]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC53":
            blocks = [7,2,3,4,5]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC75":
            blocks = [1,3,4,5,2]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC85":
            blocks = [1,2,4,5,3]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC124":
            blocks = [1,2,3]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC81":
            blocks = [1, 6, 12, 15, 19]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names  
        elif subject == "EC159":
            blocks = [6, 18] # [6, 18, 24] 最后一个block的time错误较多，暂时忽略
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2] # [1, 2, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names 
        elif subject == "EC166":
            blocks = [2, 4, 5, 6, 7]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            bursc_blocks = [1, 2, 3, 4, 5, 6]
            all_names = [bursc.get_bursc_names_for_bursc_block(block) for block in bursc_blocks]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names    
        elif subject == "EC61":
            blocks = [1, 5, 7, 11, 16]  # [29, 58, 59, 62, 69]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC92":
            blocks = [4, 5, 17, 19, 25]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        elif subject == "EC96":
            blocks = [21, 6, 11, 14, 19]
            all_hgs = [load_hg(subject, block) for block in blocks]
            all_times = [load_times(subject, block) for block in blocks]
            timit_blocks = [1, 2, 3, 4, 5]
            all_names = []
            all_names.extend([timit.get_timit_names_for_timit_block(block) for block in timit_blocks])
            return all_hgs, all_times, all_names
        
        else:
            print("not implemented for subject " + subject)
            return None

def get_asccd_names_for_asccd_block(asccd_block):
    file_path = os.path.join(asccd_data_path, "blocks", "block" + str(asccd_block) + "_stims.txt")
    with open(file_path) as f:
        names = f.readlines()
    names = [token.strip() for token in names]
    asccd_names = []
    for name in names:
        if "_" in name:
            asccd_names.append(name)
        else:
            speaker, story = asccd.get_speaker_story_from_performance(name)
            all_part_names = asccd.get_asccd_names_for_performance(speaker, story)
            asccd_names.extend(all_part_names)
    return asccd_names

def get_wavpaths_for_asccd_block(asccd_block):
    names = get_asccd_names_for_asccd_block(asccd_block)
    wavpaths = []
    for name in names:
        if "_" in name:
            speaker, _, _ = asccd.get_speaker_story_part_from_asccd_name(name)
            wavpaths.append(os.path.join(asccd_data_path, "wav", speaker + "wav", name + str(".wav")))
        else:
            wavpaths.append(os.path.join(asccd_data_path, "performance_wavs", name + ".wav"))
    return wavpaths
