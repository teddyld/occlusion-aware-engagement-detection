import os
import subprocess
import shutil
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

DAISEE_LABEL_MAP = {
    'Boredom': 0,
    'Engaged': 1,
    'Disengaged': 2,
    'Frustration': 3,
    'Confusion': 4
}

DAISEE_GENDER_MAP = {
    'Male': 0,
    'Female': 1
}

class DAiSEE(Dataset):
    def __init__(self, ttv, transform=None, apply_dropout_tf=False):
        """
        Arguments:
            ttv (string): 'Train', 'Validation', or 'Test'
            transform (callable, optional): Optional transform to be applied on image
            apply_dropout_tf (boolean, optional): Optional boolean that if asserted True will apply the *Dropout transforms to the image
        """
        if ttv not in ['Train', 'Validation', 'Test']:
            raise ValueError(f"ttv={ttv} is not 'Train', 'Validation', 'Test'") 

        self.data = os.path.join(config.DAISEE_DATA_PATH, ttv)
        self.labels = np.load(os.path.join(config.DAISEE_ANNOTATIONS_PATH, ttv + '_labels.npy'))
        self.landmarks = np.load(os.path.join(config.DAISEE_ANNOTATIONS_PATH, ttv + '_landmarks.npy'))
        self.genders = np.load(os.path.join(config.DAISEE_ANNOTATIONS_PATH, ttv + '_genders.npy'))
        self.transform = transform
        self.ttv = ttv
        self.apply_dropout_tf = apply_dropout_tf

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = self.ttv + '_' + str(idx) + '.jpg'
        image = cv2.imread(os.path.join(self.data, image_name), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            if self.apply_dropout_tf:
                # Simulate facial accessories and external occlusions
                keypoints = self.landmarks[idx]
                dropout_tf = A.Compose([
                    A.OneOf([
                        transforms.LandmarksDropout(landmarks=keypoints, landmarks_weights=(1, 1, 1), dropout_height_range=(4, 6), dropout_width_range=(4, 6), fill_value=0),
                        transforms.ALOTDropout(num_holes_range=(1, 1), hole_height_range=(8, 24), hole_width_range=(8, 24)),
                        transforms.EdgeDropout(edge_height_range=(8, 16), edge_width_range=(8, 16), fill_value=0),
                    ], p=0.5)
                ])
                image = dropout_tf(image=image)['image']
            image = self.transform(image=image)['image']
    
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        return image, label
    
    # Debugging/Helper Methods
    def get_labels(self):
        return self.labels
    
    def get_landmarks(self):
        return self.landmarks
    
    def get_genders(self):
        return self.genders
    
def extract_frames(ttv, verbose=True):
    """
    Extracts frames from the DAiSEE dataset using ffmpeg.
    Arguments:
        ttv (string): Data split to extract i.e. one of 'Train', 'Validation', or 'Test'
        verbose (boolean, optional): Optional boolean that if asserted True will print the frequency of each valid class and the number of invalid entries  
    """
    if ttv not in ['Train', 'Validation', 'Test']:
        raise ValueError(f"ttv={ttv} is not 'Train', 'Validation', 'Test'") 
    
    labels = pd.read_csv(os.path.join(config.DAISEE_ANNOTATIONS_PATH, f"{ttv}Labels.csv"))
    
    valid_clips = get_valid_clips(labels)
    users = os.listdir(f'{config.DAISEE_DATA_PATH}/{ttv}')
    
    missing_clips = []

    pbar = tqdm(users, leave=True)
    for user in pbar:
        pbar.set_description(f"Writing {ttv} dataset to disk")
        currUser_path = os.path.join(config.DAISEE_DATA_PATH, ttv, user)
        currUser = os.listdir(currUser_path)
        for extract in currUser:
            try:
                clip = os.listdir(f'{currUser_path}/{extract}')[0]
                src_path = os.path.join(currUser_path, extract) + '/'
                dest_path = os.path.join(config.DAISEE_DATA_PATH, ttv)
                if clip in valid_clips:  
                    split_video(clip, clip[:-4], src_path, dest_path)
                # Remove clip folder
                shutil.rmtree(src_path)
            except IndexError:
                missing_clips.append(extract)
                print(f'Index does not exist at user {user} at clip {extract}')
        # Remove user folder
        shutil.rmtree(currUser_path)
    
    out_labels = []
    out_landmarks = []
    out_genders = []
    landmark_found = 0
    all_files = os.listdir(f'{config.DAISEE_DATA_PATH}/{ttv}')
    pbar = tqdm(all_files, leave=True)
    for img_name in pbar:
        pbar.set_description(f"Writing {ttv} landmarks, labels, and genders to disk")
        image_path = os.path.join(config.DAISEE_DATA_PATH, ttv, img_name)
        
        # Pre-processing face detection step
        face_features = detect.detect_face_features(image_path)
        
        if face_features:
            # Append to labels
            label = get_image_frame_label(img_name, labels)
            out_labels.append(label)
            gender = get_image_frame_gender(img_name)
            out_genders.append(gender)
            
            bbox, landmarks = face_features
            image = cv2.imread(image_path)
            # Crop to image bbox region
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # Resize image to 48x48 resolution
            image = cv2.resize(image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
            # Overwrite image
            new_image_path = os.path.join(config.DAISEE_DATA_PATH, ttv, str(ttv) + '_' + str(landmark_found) + '.jpg')
            cv2.imwrite(new_image_path, image)
            
            # Append to landmarks annotations
            out_landmarks.append(landmarks)
            landmark_found += 1

        # Remove original image from disk
        os.remove(image_path)
    
    # Write labels to disk
    np.save(os.path.join(config.DAISEE_ANNOTATIONS_PATH, str(ttv) + '_' + 'labels.npy'), out_labels)
    
    # Write landmarks to disk
    np.save(os.path.join(config.DAISEE_ANNOTATIONS_PATH, str(ttv) + '_' + 'landmarks.npy'), out_landmarks)
    
    # Write genders to disk
    np.save(os.path.join(config.DAISEE_ANNOTATIONS_PATH, str(ttv) + '_' + 'genders.npy'), out_genders)
    
    if verbose:
        print(f"CompreFace succesfully detected a face {(landmark_found / len(all_files)) * 100}% of times in the {ttv} set")
        ttv_distribution = print_dataset_statistics(ttv, labels, valid_clips, missing_clips)
        return ttv_distribution

def split_video(video_file, image_name_prefix, src_path, destination_path):
    """
    Split the 10 second video clip into 10 frames, 1 frame/second    
    """
    return subprocess.run('ffmpeg -i "' + src_path+video_file + '" -r 1 -vframes 10 ' + image_name_prefix + '_%d.jpg -hide_banner', shell=True, cwd=destination_path, capture_output=True)

def get_valid_clips(labels):
    """
    Returns a list of only the valid clips from the DAiSEE label file.
    Valid clips are those with non-ambiguous multi-class labelling because of the way the dataset labels have been aggregated.
    Arguments:
        labels (DataFrame): DataFrame for the DAiSEE csv labels file
    """
    valid_clips = []
    for _, row in labels.iterrows(): 
        boredom = row['Boredom']
        engagement = row['Engagement']
        frustration = row['Frustration']
        confusion = row['Confusion']
        
        if all(boredom > i for i in [engagement, frustration, confusion]) or all(engagement > i for i in [boredom, frustration, confusion]) or all(frustration > i for i in [boredom, engagement, confusion]) or all(confusion > i for i in [boredom, engagement, frustration]):
            valid_clips.append(row['ClipID'])

    return valid_clips

def get_image_frame_label(img_name, labels):
    """
    Return the label of the image frame
    Arguments:
        img_name (string): Defines the image frame which is structured like [CLIPID]_[FRAME_NUMBER].jpg
        labels (DataFrame): DataFrame for the DAiSEE csv labels file
    """
    # Get the entry of the image in the labels DataFrame
    clip_id = img_name.split("_")[0]
    row = labels[labels['ClipID'].isin([clip_id + ".mp4", clip_id + ".avi"])]

    boredom = row['Boredom'].item()
    engagement = row['Engagement'].item()
    frustration = row['Frustration'].item()
    confusion = row['Confusion'].item()
    
    if all(boredom > i for i in [engagement, frustration, confusion]):
        return DAISEE_LABEL_MAP['Boredom']
    elif all(engagement > i for i in [boredom, frustration, confusion]):
        if engagement < 2:
            return DAISEE_LABEL_MAP['Disengaged']
        else:
            return DAISEE_LABEL_MAP['Engaged']
    elif all(frustration > i for i in [boredom, engagement, confusion]):
        return DAISEE_LABEL_MAP['Frustration']
    elif all(confusion > i for i in [boredom, engagement, frustration]):
        return DAISEE_LABEL_MAP['Confusion']

def get_image_frame_gender(img_name):
    """
    Return the gender of the image frame subject
    Arguments:
        img_name (string): Defines the image frame which is structured like [CLIPID]_[FRAME_NUMBER].jpg
    """
    clip_id = img_name.split("_")[0]
    
    # Open stream of video clips containing male subjects
    with open(f'{config.DAISEE_GENDER_PATH}/Males.txt', 'r') as males:
        content = males.read()
        if clip_id + ".avi" in content or clip_id + ".mp4" in content:
            return DAISEE_GENDER_MAP['Male']
        else:
            return DAISEE_GENDER_MAP['Female']


def print_dataset_statistics(ttv, labels, valid_clips, missing_clips):
    """
    Print the statistics from the frame extraction process including number of invalid and valid clips (after parsing out missing clips) and class distribution of clips
    Arguments:
        ttv (string): Data split extracted i.e. one of 'Train', 'Validation', or 'Test'
        labels (DataFrame): DataFrame for the DAiSEE csv labels file
        valid_clips (List[str]): Defines a list of valid clips (not accounting missing clips)
        missing_clips (List[str]): Defines a list of missing clips
    """

    daisee_frequency = {
        'Boredom': 0,
        'Engaged': 0,
        'Disengaged': 0,
        'Frustration': 0,
        'Confusion': 0,
    }
    
    # Account for missing clips in the valid_clips list
    for clip in valid_clips:
        # Remove the missing clip
        if clip[:-4] in missing_clips:
            valid_clips.remove(clip)
    
    total_samples = 0
    total_males = 0
    total_females = 0
    for _, row in labels.iterrows():
        boredom = row['Boredom']
        engagement = row['Engagement']
        frustration = row['Frustration']
        confusion = row['Confusion']
        
        clip = row['ClipID']
        
        # Count class distribution of valid clips
        if clip in valid_clips:
            if all(boredom > i for i in [engagement, frustration, confusion]):
                daisee_frequency['Boredom'] += 1
            elif all(engagement > i for i in [boredom, frustration, confusion]):
                if engagement < 2:
                    daisee_frequency['Disengaged'] += 1
                else:
                    daisee_frequency['Engaged'] += 1
            elif all(frustration > i for i in [boredom, engagement, confusion]):
                daisee_frequency['Frustration'] += 1
            elif all(confusion > i for i in [boredom, engagement, frustration]):
                daisee_frequency['Confusion'] += 1

            with open(f'{config.DAISEE_GENDER_PATH}/Males.txt', 'r') as males:
                content = males.read()
                if clip in content:
                    total_males += 1
                else:
                    total_females += 1

        total_samples += 1

    print(f"-- {ttv} Split Statistics --")
    print(f"Number of males = {total_males}, Number of females = {total_females}")
    print(f"Total number of clips = {total_samples}")
    print(f"  Number of invalid clips = {total_samples - len(valid_clips)}")
    print(f"  Number of valid clips = {len(valid_clips)}")
    print(f"DAiSEE clip class distribution: {daisee_frequency}")
    print("---------------------------\n")
    return daisee_frequency

def get_datasets(augment_tf=None, apply_dropout_tf=False):
    """
    Return train, validation, and test datasets with optional 'augment_tf' transform to apply to train dataset
    """
    train_dataset = DAiSEE('Train', augment_tf, apply_dropout_tf)
    valid_dataset = DAiSEE('Validation', transforms.simple_tf)
    test_dataset = DAiSEE('Test', transforms.simple_tf)
    return train_dataset, valid_dataset, test_dataset

def get_dataloaders(augment_tf=None, bs=64, apply_dropout_tf=False):
    """
    Return train, valid, and test DataLoaders with 'augment_tf' transform to apply to training set and batch size 'bs'. If apply_dropout_tf is True, then the image will also be augmented with *Dropout transforms
    """
    train, valid, test = get_datasets(augment_tf, apply_dropout_tf)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test, batch_size=bs, shuffle=True)
    return train_loader, valid_loader, test_loader
