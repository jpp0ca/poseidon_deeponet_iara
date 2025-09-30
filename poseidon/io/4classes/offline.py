"""
    Projeto Marinha do Brasil

    Autor: Pedro Henrique Braga Lisboa (pedro.lisboa@lps.ufrj.br)
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
from __future__ import print_function, division

import os
import h5py
import warnings
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

def load_raw_data(input_db_path, verbose=0):
    """
        Loads sonar audio datafiles on memory. 

        This function returns a nested hashmap associating each run audio data with its
        class and filename. The audio information is composed by 
        the frames stored in a numpy array and the file informed sample rate.
        
        E.g. for database '4classes' the returned dictionary will be set like:
        
        ClassA:
            navio10.wav: 
                signal: np.array
                sample_rate: np.float64
            navio11.wav: 
                signal: np.array
                sample_rate: np.float64
        ClassB:
            navio20.wav: 
                ...
            navio21.wav:
                ...
            ...
        ...
            
        params:
            input_data_path (string): 
                path to database folder
        return (SonarDict): 
                nested dicionary in which the basic unit contains
                a record of the audio (signal key) in np.array format
                and the sample_rate (fs key) stored in floating point. 
                The returned object also contains a method for applying
                functions over the runs (see SonarDict.apply).
                the map is made associating each tuple to the corresponding
                name of the run (e.g. )
    """

    if verbose:
        print('Reading Raw data in path %s' % input_db_path)

    class_folders = [folder for folder in os.listdir(input_db_path)
                        if not folder.startswith('.')]
    raw_data = dict()
    
    for cls_folder in class_folders:
        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        if not runfiles:  # No files found inside the class folder
            if verbose:
                print('Empty directory %s' % cls_folder)
            continue
        if verbose:
            print('Reading %s' % cls_folder)

        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        runpaths = [os.path.join(input_db_path, cls_folder, runfile)
                    for runfile in runfiles]
        runfiles = [runfile.replace('.wav', '') for runfile in runfiles]

        audio_data = [read_audio_file(runpath) for runpath in runpaths]
        raw_data[cls_folder] = {
            runfile: {'signal': signal, 'fs': fs}
            for runfile, (signal, fs)  in zip(runfiles, audio_data)
        }

    return SonarDict(raw_data)

# class RunRecord(dict):
#     """
#     Basic dicionary for storing the runs
#     binding the data with its respective metadata(sample rate)
#     This wrapper was made to standardize the keynames. 
#     """

#     def __init__(self, signal, fs):
#         self.__dict__['signal'] = signal
#         self.__dict__['fs'] = fs

#     def __getitem__(self , k):
# 	return self.__dict__[k]

class SonarDict(dict):
    """ 
    Wrapper for easy application of preprocessing functions 
    """
    def __init__(self, raw_data):
        super(SonarDict, self).__init__(raw_data)

    @staticmethod
    def from_hdf5(filepath):
        f = h5py.File(filepath, 'r')
        raw_data = SonarDict.__level_from_hdf5(f)
        f.close()
        return SonarDict(raw_data)
        
    @staticmethod
    def __level_from_hdf5(group_level):
        level_dict = dict()
        for key in group_level.keys():
            if isinstance(group_level[key], h5py._hl.group.Group):
                level_dict[key] = SonarDict.__level_from_hdf5(group_level[key])
            elif isinstance(group_level[key], h5py._hl.dataset.Dataset):
                # if isinstance(group_level[key].dtype, 'float64')
                level_dict[key] = group_level[key][()]
            else:
                raise ValueError

        return level_dict


    def to_hdf5(self, filepath):
        f = h5py.File(filepath, 'w')
        SonarDict.__level_to_hdf5(self, f, '')
        f.close()

    @staticmethod
    def __level_to_hdf5(dictionary_level, f, dpath):
        for key in dictionary_level.keys():
            ndpath = dpath + '/%s' % key
            if isinstance(dictionary_level[key], dict):
                SonarDict.__level_to_hdf5(dictionary_level[key], f, ndpath)
            else:
                if isinstance(dictionary_level[key], np.ndarray):
                    dtype = dictionary_level[key].dtype
                else:
                    dtype = type(dictionary_level[key])
                f.create_dataset(ndpath, data=dictionary_level[key], dtype=dtype)

    def apply(self, fn,*args, **kwargs):
        """ 
        Apply a function over each run of the dataset.

        params:
            fn: callable to be applied over the data. Receives at least
                one parameter: dictionary (RunRecord)
            args: optional params to fn
            kwargs: optional named params to fn
        
        return:
            new SonarDict object with the processed data. The inner structure
            of signal, sample_rate pair is mantained, which allows for chaining
            several preprocessing steps.

        """
        sonar_cp = self.copy()

        return SonarDict({
            cls_name: self._apply_on_class(cls_data, fn, *args, **kwargs) 
            for cls_name, cls_data in sonar_cp.items()
        })

    def _apply_on_class(self, cls_data, fn, *args, **kwargs):
        """
        Apply a function over each run signal of a single class.
        Auxiliary function for applying over the dataset
        """
        return {
            run_name: fn(raw_data, *args, **kwargs)
            for run_name, raw_data in cls_data.items()
        }

def read_audio_file(filepath):
    signal, fs = sf.read(filepath)

    return signal, fs

