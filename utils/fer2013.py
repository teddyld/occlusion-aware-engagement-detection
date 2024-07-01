import os
import pandas as pd
import numpy as np
import utils.config as config
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import utils.transforms as transforms
import utils.detect as detect
from tqdm import tqdm
import albumentations as A

class FER2013(Dataset):
    def __init__(self, ttv, transform=None, apply_landmark_tf=False):
        '''
        Arguments:
            ttv (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on image
            apply_landmark_tf (boolean, optional): Optional boolean that if asserted True will apply the LandmarksDropout transform to the image
        '''
        if ttv not in ['train', 'val', 'test']:
            raise ValueError(f"ttv={ttv} is not 'train', 'val', 'test'") 
        self.data = os.path.join(config.DATA_PATH, ttv)
        self.labels = np.load(os.path.join(config.ANNOTATIONS_PATH, ttv + '_labels.npy'))
        self.landmarks = np.load(os.path.join(config.ANNOTATIONS_PATH, ttv + '_landmarks.npy'))
        self.transform = transform
        self.ttv = ttv
        self.apply_landmark_tf = apply_landmark_tf
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = self.ttv + '_' + str(idx) + '.jpg'
        image = cv2.imread(os.path.join(self.data, image_name), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            if self.apply_landmark_tf:
                # Simulate facial accessories and external occlusions
                keypoints = self.landmarks[idx]
                landmark_tf = A.Compose([
                    transforms.LandmarksDropout(landmarks=keypoints, landmarks_weights=(1, 1, 1), dropout_height_range=(4, 4), dropout_width_range=(4, 4), fill_value="random")
                ])
                image = landmark_tf(image=image)['image']
            image = self.transform(image=image)['image']
    
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        return image, label
    
    # Debugging/Helper Methods
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

def parse_data(data_split):
    '''
    Write data in Dataframe to disk, optionally apply preprocessing to image
    Arguments:
        data_split (dictionary) - dictionary containing key-value pairs of dataset split (string) and Dataframe of split
    '''
    
    for ttv, data in data_split.items():
        images = data[' pixels'].tolist()
        labels = data['emotion'].tolist()
        out_labels = []
        out_landmarks = []

        # Write images to disk
        image_idx = 0
        loop = tqdm(enumerate(images), total=len(images), leave=True)
        for idx, image in loop:
            loop.set_description(f'Writing {ttv} dataset to disk')
            
            # Write image to disk
            image_path = os.path.join(config.DATA_PATH, ttv, str(ttv) + '_' + str(image_idx) + '.jpg')
            cv2.imwrite(image_path, image)
            out_labels.append(labels[idx])
            image_idx += 1

            # Pre-processing face detection step
            face_features = detect.detect_face_features(image_path)
            
            if face_features:
                bbox, landmarks = face_features
                image = cv2.imread(image_path)
                # Crop to image bbox region
                image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # Resize image to 48x48 resolution
                image = cv2.resize(image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                # Overwrite image
                cv2.imwrite(image_path, image)
                # Append to landmarks annotations
                out_landmarks.append(landmarks)
            else:
                out_landmarks.append([[0, 0] for _ in range(5)])

        # Write labels to disk
        np.save(os.path.join(config.DATA_PATH, 'annotations', str(ttv) + '_' + 'labels.npy'), out_labels)
        
        # Write landmarks to disk
        np.save(os.path.join(config.DATA_PATH, 'annotations', str(ttv) + '_' + 'landmarks.npy'), out_landmarks)

def get_datasets(augment_tf=None, apply_landmark_tf=False):
    '''
    Return train, validation, and test datasets with optional 'augment_tf' transform to apply to train dataset
    '''
    train_dataset = FER2013('train', augment_tf, apply_landmark_tf)
    valid_dataset = FER2013('val', transforms.simple_tf)
    test_dataset = FER2013('test', transforms.simple_tf)
    return train_dataset, valid_dataset, test_dataset

def get_dataloaders(augment_tf=None, bs=64, apply_landmark_tf=False):
    '''
    Return train, valid, and test DataLoaders with 'augment_tf' transform to apply to training set and batch size 'bs'. If apply_landmark_tf is True, then the image will also be augmented with LandmarksDropout
    '''
    train, valid, test = get_datasets(augment_tf, apply_landmark_tf)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test, batch_size=bs, shuffle=True)
    return train_loader, valid_loader, test_loader
