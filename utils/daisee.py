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
import seaborn as sns
import matplotlib.pyplot as plt

DAISEE_LABEL_TO_ONE_HOT = {
    0: [1, 0, 0, 0],
    1: [0, 1, 0, 0],
    2: [0, 0, 1, 0],
    3: [0, 0, 0, 1]
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
        gender = torch.tensor(self.genders[idx], dtype=torch.uint8)
        return image, label, gender
    
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
    
    users = os.listdir(f'{config.DAISEE_DATA_PATH}/{ttv}')
    
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
                if clip_has_valid_label(clip, labels):
                    split_video(clip, clip[:-4], src_path, dest_path)
                # Remove clip folder
                shutil.rmtree(src_path)
            except IndexError:
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
        plot_dataset_statistics(ttv)

def split_video(video_file, image_name_prefix, src_path, destination_path):
    """
    Split the 10 second video clip into 10 frames, 1 frame/second    
    """
    return subprocess.run('ffmpeg -i "' + src_path+video_file + '" -r 1 -vframes 10 ' + image_name_prefix + '_%d.jpg -hide_banner', shell=True, cwd=destination_path, capture_output=True)

def clip_has_valid_label(clip_id, labels):
    """
    Returns False if the clip is not a valid label in the labels DataFrame, otherwise True
    Arguments:
        clip_id (string): Defines the clip ID which is structured like [CLIPID].mp4/.avi
        labels (DataFrame): DataFrame for the DAiSEE csv labels file
    """
    row = labels[labels['ClipID'] == clip_id]
    if row.empty:
        return False
    return True

def get_image_frame_label(img_name, labels):
    """
    Return the label of the image frame, where each label is a one-hot encoded vector of size 16. For example, 
            [0, 1, 0, 0, | 0, 0, 1, 0, | 0, 0, 0, 1, | 1, 0, 0, 0] 
    represents the labels for low Boredom, high Engagement, very high Confusion, and very low Frustration respectively.
    Arguments:
        img_name (string): Defines the image frame which is structured like [CLIPID]_[FRAME_NUMBER].jpg
        labels (DataFrame): DataFrame for the DAiSEE csv labels file
    """
    # Get the entry of the image in the labels DataFrame
    clip_id = img_name.split("_")[0]
    row = labels[labels['ClipID'].isin([clip_id + ".mp4", clip_id + ".avi"])]

    boredom = row['Boredom'].item()
    engagement = row['Engagement'].item()
    confusion = row['Confusion'].item()
    frustration = row['Frustration'].item()
 
    return DAISEE_LABEL_TO_ONE_HOT[boredom] + DAISEE_LABEL_TO_ONE_HOT[engagement] + DAISEE_LABEL_TO_ONE_HOT[confusion] + DAISEE_LABEL_TO_ONE_HOT[frustration]

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


def plot_dataset_statistics(ttv):
    """
    Plot the statistics from the frame extraction process
    Arguments:
        ttv (string): Data split extracted i.e. one of 'Train', 'Validation', or 'Test'
    """
    
    # Gender plot
    genders = np.load(os.path.join(config.DAISEE_ANNOTATIONS_PATH, ttv + '_genders.npy'))

    genders_plt = sns.countplot(x=genders, order=[0, 1], width=0.5, hue=genders, legend=False)
    genders_plt.set_title("Gender Distribution in " + ttv + " Dataset")
    genders_plt.set_xlabel("Gender")
    genders_plt.set_ylabel("Number of image frames")
    genders_plt.set_xticks([0, 1])
    genders_plt.set_xticklabels(["Male", "Female"])
    
    for c in genders_plt.containers:
        genders_plt.bar_label(c, fmt=lambda x: f"{x:0.0f}" if x > 0 else "")

    plt.show()
    
    plot_label_frequency(ttv)    

def plot_label_frequency(ttv, benchmark=None):
    """
    Plot the frequency of each label in the dataset or only for the benchmark metric label
    """
    # Labels plot
    labels = np.load(os.path.join(config.DAISEE_ANNOTATIONS_PATH, ttv + '_labels.npy'))
    
    frequency_map = {
        "Boredom": {
            "Very Low": 0,
            "Low": 0,
            "High": 0,
            "Very High": 0
        },
        "Engagement": {
            "Very Low": 0,
            "Low": 0,
            "High": 0,
            "Very High": 0
        },
        "Confusion": {
            "Very Low": 0,
            "Low": 0,
            "High": 0,
            "Very High": 0
        },
        "Frustration": {
            "Very Low": 0,
            "Low": 0,
            "High": 0,
            "Very High": 0
        },
    }
    
    for label in labels:
        frequency_map["Boredom"]["Very Low"] += label[0]
        frequency_map["Boredom"]["Low"] += label[1]
        frequency_map["Boredom"]["High"] += label[2]
        frequency_map["Boredom"]["Very High"] += label[3]
        frequency_map["Engagement"]["Very Low"] += label[4]
        frequency_map["Engagement"]["Low"] += label[5]
        frequency_map["Engagement"]["High"] += label[6]
        frequency_map["Engagement"]["Very High"] += label[7]
        frequency_map["Confusion"]["Very Low"] += label[8]
        frequency_map["Confusion"]["Low"] += label[9]
        frequency_map["Confusion"]["High"] += label[10]
        frequency_map["Confusion"]["Very High"] += label[11]
        frequency_map["Frustration"]["Very Low"] += label[12]
        frequency_map["Frustration"]["Low"] += label[13]
        frequency_map["Frustration"]["High"] += label[14]
        frequency_map["Frustration"]["Very High"] += label[15]
    
    if benchmark: 
        frequency_map = frequency_map[benchmark]
    df = pd.DataFrame.from_dict(frequency_map, orient='index')
    color = sns.color_palette("viridis", 4)
    
    labels_plt = df.plot(kind='bar', rot=0, title=f"Label Distribution in {ttv} Dataset", xlabel=f"{benchmark} Label" if benchmark else "Label", ylabel="Number of image frames", width=0.85, color=color, figsize=(10, 5), legend=False if benchmark else True)
    
    for c in labels_plt.containers:
        labels_plt.bar_label(c, fmt=lambda x: f"{x:0.0f}" if x > 0 else "")
        
    plt.show()

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
