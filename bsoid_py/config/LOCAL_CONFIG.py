BASE_PATH = '/Users/ahsu/B-SOID/datasets'  # Base directory path.

TRAIN_FOLDERS = ['/Train']  # Folder paths containing training data.
PREDICT_FOLDERS = ['/041919', '/042219']  # Folder paths containing new dataset to predict using built classifier.

# This version requires the six body parts Snout/Head, Forepaws/Shoulders, Hindpaws/Hips, Tailbase.
BODYPARTS = {
    'Snout/Head': 0,
    'Neck': None,
    'Forepaw/Shoulder1': 1,
    'Forepaw/Shoulder2': 2,
    'Bodycenter': None,
    'Hindpaw/Hip1': 3,
    'Hindpaw/Hip2': 4,
    'Tailbase': 5,
    'Tailroot': None
}

FPS = 60  # Frame-rate of your video,
# note that you can use a different number for new data as long as the video is same scale/view
COMP = 1  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.

# THINGS YOU DEFINITELY NEED TO CHANGE

# Output directory to where you want the analysis to be stored
OUTPUT_PATH = '/Users/ahsu/Desktop/'
# Machine learning model name
MODEL_NAME = 'c57bl6_n2_120min'
# Pick one Machine learning model you'd like to use
FINALMODEL_NAME = 'bsoid_py_beta/bsoid_c57bl6_n2_120min_20200413_1941.sav'

# Pick a video
VID_NAME = '/Users/ahsu/B-SOID/datasets/041919/2019-04-19_09-34-36cut0_30min.mp4'
# What number would be video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)
ID = 0
# Create a folder to store extracted images, make sure this folder exist.
# This program will predict labels and print them on these images
FRAME_DIR = '/Users/ahsu/B-SOID/datasets/041919/0_30min_10fpsPNGs'
# In addition, this will also create an entire sample group videos for ease of understanding
SHORTVID_DIR = '/Users/ahsu/B-SOID/datasets/041919/examples'
