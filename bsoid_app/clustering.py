import os

import hdbscan
import joblib
#import numpy as np
import cupy as np
from bsoid_app.config import *
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_workspace import load_clusters


class Cluster:

    def __init__(self, working_dir, prefix, sampled_embeddings, min_=0.5, max_=1.0):
        self.working_dir = working_dir
        self.prefix = prefix
        self.sampled_embeddings = sampled_embeddings
        self.cluster_range = [min_, max_]
        self.min_cluster_size = []
        self.assignments = []
        self.assign_prob = []
        self.soft_assignments = []

    def hierarchy(self):
        print(str.join('', ('Identifying... Here is a random fact: ', '')))
        max_num_clusters = -np.infty
        num_clusters = []
        self.min_cluster_size = np.linspace(self.cluster_range[0], self.cluster_range[1], 25)
        for min_c in self.min_cluster_size:
            learned_hierarchy = hdbscan.HDBSCAN(
                prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * self.sampled_embeddings.shape[0])),
                **HDBSCAN_PARAMS).fit(self.sampled_embeddings)
            num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
            if num_clusters[-1] > max_num_clusters:
                max_num_clusters = num_clusters[-1]
                retained_hierarchy = learned_hierarchy
        self.assignments = retained_hierarchy.labels_
        self.assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
        self.soft_assignments = np.argmax(self.assign_prob, axis=1)
        print('Done assigning labels for **{}** instances ({} minutes) '
              'in **{}** D space'.format(self.assignments.shape,
                                         round(self.assignments.shape[0] / 600),
                                         self.sampled_embeddings.shape[1]))

    def save(self):
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_clusters.sav'))), 'wb') as f:
            joblib.dump([self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments], f)

    def main(self):
        try:
            [self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments] = \
                load_clusters(self.working_dir, self.prefix)
            print(
                '**_CHECK POINT_**: Done assigning labels for **{}** instances in **{}** D space. Move on to __create '
                'a model__.'.format(self.assignments.shape, self.sampled_embeddings.shape[1]))
            print('Your last saved run range was __{}%__ to __{}%__'.format(self.min_cluster_size[0],
                                                                            self.min_cluster_size[-1]))
            self.hierarchy()
            self.save()
        except (AttributeError, FileNotFoundError) as e:
            self.hierarchy()
            self.save()
