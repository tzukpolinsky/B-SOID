import glob
import os.path
import sys
from datetime import date

import numpy as np
import pandas as pd

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, predict
from bsoid_app.bsoid_utilities.likelihoodprocessing import adp_filt
from bsoid_app.bsoid_utilities.load_workspace import *
import json

from bsoid_app.bsoid_utilities.load_workspace import load_feats


def compile_single_file(data_file, working_dir, prefix, framerate, root_path):
    pose_chosen = []
    file0_df = pd.read_csv(data_file, low_memory=False)
    file0_array = np.array(file0_df)
    raw_input_data = []
    sub_threshold = []
    processed_input_data = []
    input_filenames = []
    p = file0_array[0, 1:-1:3]
    for a in p:
        index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
        if index not in pose_chosen:
            pose_chosen += index
    pose_chosen.sort()
    if len(pose_chosen) > 0:
        file_j_df = pd.read_csv(data_file, low_memory=False)
        file_j_processed, p_sub_threshold = adp_filt(file_j_df, pose_chosen)
        raw_input_data.append(file_j_df)
        sub_threshold.append(p_sub_threshold)
        processed_input_data.append(file_j_processed)
        input_filenames.append(data_file)
    with open(os.path.join(working_dir, str.join('', (prefix, '_data.sav'))), 'wb') as f:
        joblib.dump(
            [root_path, [], framerate, pose_chosen, input_filenames,
             raw_input_data, processed_input_data, sub_threshold], f)
    return [root_path, [], framerate, pose_chosen, input_filenames,
            raw_input_data, processed_input_data, sub_threshold]


def file_by_file(settings):
    print("start bsoid server")
    glob_query = settings["data_path"] + '/*.' + settings["file_type"]
    data_files = glob.glob(glob_query)
    # processor = data_preprocess.preprocess(settings["software"], settings["file_type"], settings["data_path"],
    #                                        int(settings["framerate"]), working_dir)
    # root_path, data_directories, framerate, pose_chosen, input_filenames, raw_input_data, processed_input_data, sub_threshold = processor.compile_data()

    for data_file in data_files:
        if not os.path.isfile(data_file):
            continue
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")
        prefix = d4 + data_file.split("/")[0]
        working_dir = settings["data_path"] + "/" + prefix
        os.mkdir(working_dir)
        root_path, data_directories, framerate, pose_chosen, input_filenames, raw_input_data, processed_input_data, sub_threshold = compile_single_file(
            os.path.join(settings["data_path"], data_file), working_dir, prefix, int(settings["framerate"]),
            settings["data_path"])

        extractor = extract_features.Extract(working_dir, prefix, processed_input_data, framerate,
                                             float(settings["train_size"]))
        [sampled_features, sampled_embeddings] = extractor.main()
        cluster = clustering.Cluster(working_dir, prefix, sampled_embeddings)
        cluster.main()
        [_, assignments, assign_prob, soft_assignments] = load_clusters(working_dir, prefix)
        exporter = export_training.Export(working_dir, prefix, sampled_features,
                                          assignments, assign_prob, soft_assignments)
        exporter.save_csv()
        [features, _] = load_feats(working_dir, prefix)
        learning_protocol = machine_learner.Protocol(working_dir, prefix, features, sampled_features, assignments)
        learning_protocol.main()
        [_, _, _, clf, _, predictions] = load_classifier(working_dir, prefix)
        # creator = video_creator.Creator(root_path, data_directories, processed_input_data, pose_chosen,
        #                                 working_dir, prefix, framerate, clf, input_filenames)
        # creator.main()
        predictor = predict.Prediction(root_path, data_directories, input_filenames, processed_input_data,
                                       working_dir,
                                       prefix, framerate, pose_chosen, predictions, clf)
        predictor.main()
        print("done with {}".format(data_file))
    print("done bsoid")


def main(settings):
    print("start bsoid server")
    working_dir = settings["working_dir"]

    processor = data_preprocess.preprocess(settings["software"], settings["file_type"], settings["data_path"],
                                           int(settings["framerate"]), working_dir)
    root_path, data_directories, framerate, pose_chosen, input_filenames, raw_input_data, processed_input_data, sub_threshold = processor.compile_data()
    prefix = processor.prefix
    extractor = extract_features.Extract(working_dir, prefix, processed_input_data, framerate,
                                         float(settings["train_size"]))
    [sampled_features, sampled_embeddings] = extractor.main()
    cluster = clustering.Cluster(working_dir, prefix, sampled_embeddings)
    cluster.main()
    [_, assignments, assign_prob, soft_assignments] = load_clusters(working_dir, prefix)
    exporter = export_training.Export(working_dir, prefix, sampled_features,
                                      assignments, assign_prob, soft_assignments)
    exporter.save_csv()
    [features, _] = load_feats(working_dir, prefix)
    learning_protocol = machine_learner.Protocol(working_dir, prefix, features, sampled_features, assignments)
    learning_protocol.main()
    [_, _, _, clf, _, predictions] = load_classifier(working_dir, prefix)
    # creator = video_creator.Creator(root_path, data_directories, processed_input_data, pose_chosen,
    #                                 working_dir, prefix, framerate, clf, input_filenames)
    # creator.main()
    predictor = predict.Prediction(root_path, data_directories, input_filenames, processed_input_data, working_dir,
                                   prefix, framerate, pose_chosen, predictions, clf)
    predictor.main()
    print("done bsoid")


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        settings = json.load(f)
    is_file_by_file = settings["is_file_by_file"]
    if is_file_by_file:
        file_by_file(settings)
    else:
        main(settings)
