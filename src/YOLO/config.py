import os

""" 
Data structure must look like:
data/
    train/
        class1/
            image1.jpg/png
            image2.jpg/png
        class2/
            image1.jpg/png
            image2.jpg/png
    val/
        class1/
            image1.jpg/png
            image2.jpg/png
        class2/
            image1.jpg/png
            image2.jpg/png
"""
DATA_PATH = os.path.abspath('../../data/FER-2013')
MODEL_TRAIN_DIRECTORY = 'train4'

# Do not modify below
TRAIN_DATA_PATH = DATA_PATH + '/train'
VAL_DATA_PATH = DATA_PATH + '/val'
MODEL_TRAIN_PATH =  os.path.abspath(f'./runs/classify/{MODEL_TRAIN_DIRECTORY}')

RESULT_PATH = MODEL_TRAIN_PATH + '/results.csv'
MODEL_PATH = MODEL_TRAIN_PATH + '/weights/last.pt'