import os
import pandas as pd
import numpy as np
import utils.config as config
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.plot import plot_class_distribution

class FER2013(Dataset):
    def __init__(self, ttv, transform=None):
        '''
        Arguments:
            ttv (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on image
        '''
        if ttv not in ['train', 'val', 'test']:
            raise ValueError(f"ttv={ttv} is not 'train', 'val', 'test'") 
        self.data = os.path.join(config.DATA_PATH, ttv)
        self.labels = np.load(os.path.join(config.ANNOTATIONS_PATH, ttv + '_labels.npy'))
        self.landmarks = np.load(os.path.join(config.ANNOTATIONS_PATH, ttv + '_landmarks.npy'))   
        self.transform = transform
        self.ttv = ttv
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = self.ttv + '_' + str(idx) + '.jpg'
        image = cv2.imread(os.path.join(self.data, image_name), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        return image, label
    
    def get_labels(self):
        return self.labels
    
    def get_landmarks(self):
        return self.landmarks
     
def stringToImage(s):
    '''
    Helper function: converts csv string of image to numpy array
    '''
    # Convert pixel values in string format to a np array
    image = np.array([int(x) for x in s.split(' ')]).reshape(48, 48)
    # Convert back to uint8
    image = np.uint8(image)
    return image

def load_data_split(path=config.CSV_PATH):
    '''
    Return train, validation, and test datasets of fer2013 csv as dictionary
    '''
    fer2013 = pd.read_csv(path)
    fer2013[' pixels'] = fer2013[' pixels'].apply(lambda s: stringToImage(s))
    return {
        'train': fer2013[fer2013[' Usage'] == 'Training'],
        'val': fer2013[fer2013[' Usage'] == 'PublicTest'],
        'test': fer2013[fer2013[' Usage'] == 'PrivateTest']
    }

def get_datasets(augment_tf=None):
    '''
    Return train, validation, and test datasets with optional augment_tf to apply to train dataset
    '''
    train_dataset = FER2013('train', augment_tf)
    base_tf = A.Compose([
        ToTensorV2(),
    ])
    valid_dataset = FER2013('val', base_tf)
    test_dataset = FER2013('test', base_tf)
    return train_dataset, valid_dataset, test_dataset

def get_dataloaders(augment_tf=None, bs=64):
    train, valid, test = get_datasets(augment_tf)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test, batch_size=bs, shuffle=True)
    return train_loader, valid_loader, test_loader

def get_class_weights(DEVICE=None):
    if not DEVICE:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _, _ = get_datasets()
    plot_class_distribution(train_dataset)
    labels = train_dataset.get_labels()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights).float().to(DEVICE)
    