import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, classification_report
import torch
import cv2
import utils.detect as detect


FER_CLASS_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear', 
    3:'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

FER_CLASS_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def plot_preprocess_step(data_split, which_ttv, process_function):
    '''
    Plot a random image from the data_split besides the image with 'process_function' applied to it
    Arguments:
        data_split (dictionary) - dictionary containing keys 'train', 'val', and 'test' with values being the DataFrame of the split
        which_ttv (string) - 'train', 'val' or 'test'
        process_function (callable) - the preprocessing function to be applied on an image
    '''
    if which_ttv not in ['train', 'val', 'test']:
        print(f'{which_ttv} is not a valid argument')
        return

    dataset = data_split[which_ttv]
    
    try:
        images = dataset[' pixels'].tolist()
        labels = dataset['emotion'].tolist()
    except Exception as e:
        print(f'Error: {e}. Check if icml_face_data.csv is correct')
        
    random_image_idx = random.randint(0, len(images))
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})

    ax[0].set_title('Original Image')
    ax[1].set_title('Pre-processed Image')
    
    fig.suptitle(f'Image: {which_ttv}_{random_image_idx}.jpg \nClass: {FER_CLASS_MAP[labels[random_image_idx]]}', y=0.8)
    
    
    image = images[random_image_idx]
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(process_function(image), cmap="gray")
    
    plt.tight_layout()
    plt.show()

def plot_augmentation(loader, which_ttv, transform):
    '''
    Plot a random image from the loader and apply 'transform' to four instances of the image, plotting the resulting augmentations
    Arguments:
        loader (DataLoader) - DataLoader of image dataset
        which_ttv (string) - prepend 'train', 'val' or 'test' to image index
        transform (callable) - an albumentations transform
    '''
    if which_ttv not in ['train', 'val', 'test']:
        print(f'{which_ttv} is not a valid argument')
        return

    dataiter = iter(loader)
    images, labels = next(dataiter)
 
    random_image_idx = random.randint(0, len(images) - 1)

    fig, axes = plt.subplots(1, 5, figsize=(10, 4), subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(f'Image: {which_ttv}_{random_image_idx}.jpg \nClass: {FER_CLASS_MAP[labels[random_image_idx].item()]}', y=0.9)
    
    # Plot original image
    original_image = images[random_image_idx].permute(1, 2, 0).numpy()
    
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title('Original Image')

    # Plot augmented image + data pre-processing
    for i, ax in enumerate(axes[1:].flat):
        augmented_image = transform(image=original_image)['image'].permute(1, 2, 0).numpy()
        ax.imshow(augmented_image, cmap="gray")
        ax.set_title(f'Augmented Image {i + 1}')
    
    plt.tight_layout()
    plt.show()

def plot_fer_dataset(loader):
    '''
    Plot eight images from a single batch of a DataLoader
    Arguments:
        loader (iterable) - a DataLoader class from PyTorch which loads the FER2013 Dataset
    '''
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})

    for images, labels in loader:
        for i, ax in enumerate(axes.flat):
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            label = int(labels[i].cpu())
            
            ax.imshow(image, cmap='gray')
            
            ax.set_title(f'{FER_CLASS_MAP[label]}')
        break

    plt.tight_layout()
    plt.show()

def plot_class_distribution(dataset):
    '''
    Plot the class distribution of the FER2013 Dataset in a seaborn countplot
    Arguments:
        dataset (class) - a PyTorch Dataset
    '''
    plot = sns.countplot(x=dataset.get_labels(), hue=dataset.get_labels(), legend=False)
    plot.set_title('Class Distribution')
    plot.set_xlabel('Class')
    plot.set_ylabel('Count')
    plot.set_xticks(range(len(FER_CLASS_LABELS)))
    plot.set_xticklabels(FER_CLASS_LABELS)

def plot_training_history(results):
    '''
    Plot training history with two plots
        1) Training accuracy vs. Validation accuracy
        2) Training loss vs. Validation loss
    Arguments:
        results (dictionary) - containing keys 'train_accuracy' and 'valid_accuracy' with value of type List
    '''
    plt.figure()
    plt.plot(results['train_accuracy'])
    plt.plot(results['valid_accuracy'])
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.figure()
    plt.plot(results['train_loss'])
    plt.plot(results['valid_loss'])
    plt.title('Training Loss vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()
    
    
def plot_confusion_matrix(results):
    '''
    Plot the confusion matrix of the training results, representing the epoch with best validation accuracy
    Arguments:
        results (dictionary) - containing keys 'y_true' and 'y_pred' with value of type List 
    '''
    cm = confusion_matrix(results['y_true'], results['y_pred'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=FER_CLASS_LABELS, yticklabels=FER_CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
def plot_predictions(model, loader, device):
    '''
    Plot model predictions with true and predicted labels of the DataLoader 'loader'
    Arguments:
        model - a PyTorch model
        loader (iterable) - a DataLoader class from PyTorch which loads the FER2013 Dataset
        device (string) - the PyTorch context to use
    '''
    model.to(device)
    model.eval()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})

    # Make prediction on test set
    with torch.no_grad():
        for inputs, true_labels in loader:
            inputs, true_labels = inputs.to(device), true_labels.to(device)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)

            for i, ax in enumerate(axes.flat):
                image = inputs[i].cpu().numpy().transpose(1, 2, 0)
                true_label = int(true_labels[i].cpu())
                pred_label = int(predicted[i].cpu())
                
                ax.imshow(image, cmap='gray')
                
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'True: {FER_CLASS_MAP[true_label]}\nPredicted: {FER_CLASS_MAP[pred_label]}', color=color)
            break # break early to test only 1 batch

    plt.tight_layout()
    plt.show()
    
def plot_image_features(image_path, bbox, landmarks):
    '''
    Plot face bbox (red) and landmarks (green) on image_path in grayscale
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    
    for x, y in landmarks:
        circle = patches.Circle((x, y), radius=1, color='lime')
        ax.add_patch(circle)
    
    rect = detect.bbox_to_rect(bbox)
    ax.add_patch(rect)
    
def display_classification_report(results):
    '''
    Display the classification report of the training results, representing the epoch with best validation accuracy
    Arguments:
        results (dictionary) - containing keys 'y_true' and 'y_pred' with value of type List 
    '''
    print('=== Classification Report ===')
    print(classification_report(results['y_true'], results['y_pred'], target_names=FER_CLASS_LABELS, digits=4))