import os

# FER-2013 Paths
DATA_PATH = os.path.abspath('./data/FER-2013')

CSV_PATH = DATA_PATH + '/icml_face_data.csv'
TRAIN_PATH = DATA_PATH + '/train'
VAL_PATH = DATA_PATH + '/val'
TEST_PATH = DATA_PATH + '/test'
ANNOTATIONS_PATH = DATA_PATH + '/annotations'

# ALOT Path
ALOT_DATA_PATH = os.path.abspath('./data/ALOT')

# DAiSEE Paths
DAISEE_DATA_PATH = os.path.abspath('./data/DAiSEE/DataSet')

DAISEE_TRAIN_PATH = DAISEE_DATA_PATH + '/Train'
DAISEE_VAL_PATH = DAISEE_DATA_PATH + '/Validation'
DAISEE_TEST_PATH = DAISEE_DATA_PATH + '/Test'
DAISEE_ANNOTATIONS_PATH = DAISEE_DATA_PATH + '/../Labels'
DAISEE_GENDER_PATH = DAISEE_DATA_PATH + '/../GenderClips'