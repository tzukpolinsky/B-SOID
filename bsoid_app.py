import sys

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, predict
from bsoid_app.bsoid_utilities.load_workspace import *
import json

from bsoid_app.bsoid_utilities.load_workspace import load_feats


def main():
    print("start bsoid server")
    with open(sys.argv[1], "r") as f:
        settings = json.load(f)
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

if __name__ == "__main__":
    main()
