import os
from datetime import date

import h5py
import joblib

from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *


class preprocess:

    def __init__(self, software, file_type, root_path, framerate, working_dir):
        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []
        self.processed_input_data = []
        self.sub_threshold = []
        self.software = software
        if self.software == 'DeepLabCut':
            self.ftype = file_type
        if self.software == 'SLEAP':
            self.ftype = 'h5'
        if self.software == 'OpenPose':
            self.ftype = 'json'
        self.root_path = root_path
        try:
            os.listdir(self.root_path)
        except FileNotFoundError:
            print("no such directory")

        self.data_directories = []
        for d in os.listdir(self.root_path):
            d2 = os.path.join(self.root_path, d)
            if os.path.isdir(d2):
                self.data_directories.append(d)
        self.framerate = framerate
        self.working_dir = working_dir
        try:
            os.listdir(self.working_dir)
        except FileNotFoundError:
            print('Cannot access working directory, was there a typo or did you forget to create one?')
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")
        self.prefix = d4

    def compile_data(self):
        glob_query = os.path.join(self.root_path , self.data_directories[0]) + '/*.'+self.ftype
        data_files = glob.glob(glob_query)
        if self.software == 'DeepLabCut' and self.ftype == 'csv':
            file0_df = pd.read_csv(data_files[0], low_memory=False)
            file0_array = np.array(file0_df)
            p = file0_array[0, 1:-1:3]
            for a in p:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if len(self.pose_chosen) > 0:
                for i, fd in enumerate(self.data_directories):  # Loop through folders
                    f = get_filenames(self.root_path, fd, self.ftype)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_csv(filename, low_memory=False)
                        file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df)
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, self.processed_input_data, self.sub_threshold], f
                    )
                print('Processed a total of **{}** .{} files, and compiled into a '
                      '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                 len(self.processed_input_data)))
        elif self.software == 'DeepLabCut' and self.ftype == 'h5':
            file0_df = pd.read_hdf(data_files[0], low_memory=False)
            p = np.array(file0_df.columns.get_level_values(1)[1:-1:3])
            for a in p:
                index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_values(1))) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if len(self.pose_chosen) > 0:
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesh5(self.root_path, fd)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_hdf(filename, low_memory=False)
                        file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df)
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, self.processed_input_data, self.sub_threshold], f
                    )
                print('Processed a total of **{}** .{} files, and compiled into a '
                      '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                 len(self.processed_input_data)))
        elif self.software == 'SLEAP' and self.ftype == 'h5':
            file0_df = h5py.File(data_files[0], 'r')
            p = np.array(file0_df['node_names'][:])
            for a in p:
                index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if len(self.pose_chosen) > 0:
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesh5(self.root_path, fd)
                    for j, filename in enumerate(f):
                        file_j_df = h5py.File(filename, 'r')
                        file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df['tracks'][:][0])
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, self.processed_input_data, self.sub_threshold], f
                    )
                print('Processed a total of **{}** .{} files, and compiled into a '
                      '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                 len(self.processed_input_data)))

        elif self.software == 'OpenPose' and self.ftype == 'json':
            file0_df = read_json_single(data_files[0])
            file0_array = np.array(file0_df)
            p = file0_array[0, 1:-1:3]
            for a in p:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if len(self.pose_chosen):
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesjson(self.root_path, fd)
                    json2csv_multi(f)
                    filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                    file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                            low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, self.processed_input_data, self.sub_threshold], f
                    )
                print('Processed a total of **{}** .{} files, and compiled into a '
                      '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                 len(self.processed_input_data)))
        return [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                self.raw_input_data, self.processed_input_data, self.sub_threshold]

    def show_bar(self):
        visuals.plot_bar(self.sub_threshold)

    # def show_data_table(self):
    #     visuals.show_data_table(self.raw_input_data, self.processed_input_data)
