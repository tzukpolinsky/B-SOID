import os

import joblib
import matplotlib.pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_workspace import load_classifier


class Protocol:

    def __init__(self, working_dir, prefix, features, sampled_features, assignments):
        self.working_dir = working_dir
        self.prefix = prefix
        self.features = features
        self.sampled_features = sampled_features
        self.assignments = assignments
        self.part = 0.2
        self.it = 10
        self.x_test = []
        self.y_test = []
        self.validate_clf = []
        self.clf = []
        self.validate_score = []
        self.predictions = []

    def randomforest(self):
        try:
            x = self.sampled_features[self.assignments >= 0, :]
            y = self.assignments[self.assignments >= 0]
            x_train, self.x_test, y_train, self.y_test = train_test_split(x, y.T, test_size=self.part, random_state=42)
            print(str.join('', ('Training random forest classifier on randomly partitioned '
                                '{}%...'.format((1 - self.part) * 100))))
            self.validate_clf = RandomForestClassifier(random_state=42,n_jobs=-1)
            self.validate_clf.fit(x_train, y_train)
            self.clf = RandomForestClassifier(random_state=42,n_jobs=-1)
            self.clf.fit(x, y.T)
            self.predictions = self.clf.predict(self.features.T)
            print('Done training random forest classifier mapping '
                  '**{}** features to **{}** assignments.'.format(self.features.T.shape, self.predictions.shape))
            self.validate_score = cross_val_score(self.validate_clf, self.x_test, self.y_test, cv=self.it, n_jobs=-1)
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_randomforest.sav'))), 'wb') as f:
                joblib.dump([self.x_test, self.y_test, self.validate_clf, self.clf,
                             self.validate_score, self.predictions], f)
        except AttributeError:
            print('Sometimes this takes a bit to update, recheck identify clusters (previous step) '
                  'and rerun this in 30 seconds.')

    # def show_confusion_matrix(self):
    #     fig = visuals.plot_confusion(self.validate_clf, self.x_test, self.y_test)

    # col1, col2 = st.beta_columns([2, 2])
    # col1.pyplot(fig[0])
    # col2.pyplot(fig[1])
    # print('To improve, either _increase_ minimum cluster size, or include _more data_')

    def show_crossval_score(self):
        fig, plt = visuals.plot_accuracy(self.validate_score)
        matplotlib.pyplot.savefig(os.path.join(self.working_dir, "crossval_score.png"))
        print('To improve, either _increase_ minimum cluster size, or include _more data_')

    def main(self):
        try:
            [self.x_test, self.y_test, self.validate_clf, self.clf, self.validate_score, self.predictions] = \
                load_classifier(self.working_dir, self.prefix)
            print('**_CHECK POINT_**: Done training random forest classifier '
                  'mapping **{}** features to **{}** assignments. Move on to '
                  '__Generate video snippets for interpretation__.'.format(self.features.shape[0],
                                                                           self.predictions.shape[0]))
            self.randomforest()
            self.show_crossval_score()
        except FileNotFoundError:
            self.randomforest()
            self.show_crossval_score()
