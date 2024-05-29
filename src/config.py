import os

# To be modified
DATA_PATH = os.path.abspath('../data/FER-2013')
TRAIN_DIRECTORY = 'train3'

# Do not modify below
TRAIN_PATH =  os.path.abspath(f'./runs/classify/{TRAIN_DIRECTORY}')

RESULT_PATH = TRAIN_PATH + '/results.csv'
MODEL_PATH = TRAIN_PATH + '/weights/last.pt'