import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, classification_report
import torch
import cv2
import utils.detect as detect
import utils.fer2013 as fer2013
import utils.daisee as daisee
import utils.transforms as transforms
import albumentations as A
import utils.loops as loops

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FER_CLASS_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear', 
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

FER_CLASS_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

DAISEE_CLASS_LABELS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

DAISEE_CLASS_LABELS_DEGREES = ['VL', 'L', 'H', 'VH']

def one_hot_label_decode(one_hot_label, benchmark):
    '''
    Return the decoded one-hot label of the DAiSEE dataset with the given benchmark metric
    '''    
    if benchmark == "Boredom":
        label = one_hot_label[:4]
    elif benchmark == "Engagement":
        label = one_hot_label[4:8]
    elif benchmark == "Confusion":
        label = one_hot_label[8:12]
    else:
        label = one_hot_label[12:]

    
    if label == [1, 0, 0, 0]:
        return "Very Low " + benchmark
    elif label == [0, 1, 0, 0]:
        return "Low " + benchmark
    elif label == [0, 0, 1, 0]:
        return "High " + benchmark
    else:
       return "Very High " + benchmark
                
def plot_augmentation(transform, dataset_name, benchmark, apply_dropout_tf=False):
    '''
    Apply 'transform' four times to a random image in the train set, plotting the resulting augmentations
    Arguments:
        transform (callable) - an albumentations transform
        dataset_name (string) - Defines the dataset to take source images from. Must be one of "FER2013" or "DAiSEE"
        benchmark (string) - Defines the benchmark metric to reference in labelling of figures. Must be one of "Boredom", "Engagement", "Confusion", "Frustration"
        apply_dropout_tf (optional, boolean) - assert True to apply *Dropout transforms to image
    '''
    if dataset_name == 'FER2013':
        train_dataset, _, _ = fer2013.get_datasets()
    else:
        train_dataset, _, _ = daisee.get_datasets()

    random_idx = random.randint(0, len(train_dataset) - 1)
    image, label, _ = train_dataset[random_idx]
    
    fig, axes = plt.subplots(1, 5, figsize=(10, 4), subplot_kw={'xticks': [], 'yticks': []})
    
    if dataset_name == 'FER2013':
        fig.suptitle(f'Image label: {FER_CLASS_MAP[label.item()]}', y=0.82, size=16)
    else:
        label = one_hot_label_decode(label.tolist(), benchmark)
        fig.suptitle(f'Image label:\n {label}', y=0.82, size=16)
    
    # Plot original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title('Original Image')

    # Plot augmented image
    for i, ax in enumerate(axes[1:].flat):
        img = image.copy()
        if apply_dropout_tf:
            keypoints = train_dataset.get_landmarks()[random_idx]
            dropout_tf = A.Compose([
                A.OneOf([
                    transforms.LandmarksDropout(landmarks=keypoints, landmarks_weights=(1, 1, 1), dropout_height_range=(4, 6), dropout_width_range=(4, 6), fill_value=0),
                    transforms.ALOTDropout(num_holes_range=(1, 1), hole_height_range=(8, 24), hole_width_range=(8, 24)),
                    transforms.EdgeDropout(edge_height_range=(8, 16), edge_width_range=(8, 16), fill_value=0),
                ], p=1)
            ])
            img = dropout_tf(image=img)['image']
        img = transform(image=img)['image'].permute(1, 2, 0).numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f'Augmented Image {i + 1}')
    
    plt.tight_layout()
    plt.show()

def plot_dataset(loader, dataset_name, benchmark):
    '''
    Plot eight images from a single batch of a DataLoader
    Arguments:
        loader (iterable) - Defines the DataLoader which loads the FER2013 or DAiSEE Dataset
        dataset_name (string) - Defines the dataset to reference in labelling of figures. Must be one of "FER2013" or "DAiSEE"
        benchmark (string) - Defines the benchmark metric to reference in labelling of figures. Must be one of "Boredom", "Engagement", "Confusion", "Frustration"
    '''
    _, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
    
    images, labels, _ = next(iter(loader))

    for i, ax in enumerate(axes.flat):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        
        ax.imshow(image, cmap='gray')
        
        if dataset_name == "FER2013":
            label = int(labels[i].cpu())
            ax.set_title(f'{FER_CLASS_MAP[label]}')
        else:
            label = one_hot_label_decode(labels[i].cpu().tolist(), benchmark)
            ax.set_title(label)


    plt.tight_layout()
    plt.show()

def plot_class_distribution(dataset_name, benchmark):
    '''
    Plot the class distribution of the FER2013 or DAiSEE train dataset in a seaborn countplot
    Arguments:
        dataset_string (string) - Defines the reference dataset. Must be one of "FER2013" or "DAiSEE"
        benchmark (string) - Defines the benchmark metric to reference in labelling of figures. Must be one of "Boredom", "Engagement", "Confusion", "Frustration"
    '''
    if dataset_name not in ['FER2013', 'DAiSEE']:
        raise ValueError(f"Error: dataset must be one of 'FER2013' or 'DAiSEE'. Got {dataset_name}")

    if dataset_name == 'FER2013':
        dataset, _, _ = fer2013.get_datasets()
        plot = sns.countplot(x=dataset.get_labels(), hue=dataset.get_labels(), legend=False)
        plot.set_title(f'{dataset_name} Class Distribution')
        plot.set_xlabel('Class')
        plot.set_ylabel('Count')
        plot.set_xticks(range(len(FER_CLASS_LABELS)))
        plot.set_xticklabels(FER_CLASS_LABELS)
        
        for c in plot.containers:
            plot.bar_label(c, fmt=lambda x: f"{x:0.0f}" if x > 0 else "")
    else:
        daisee.plot_label_frequency("Train", benchmark)

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
    
def plot_gender_history(gender_accuracy):
    '''
    Plot gender history with male and female demographic accuracy
    Arguments:
        gender_accuracy (dictionary) - containing keys 'male' and 'female' with value of type List
    '''
    if len(gender_accuracy["male"]) == 0 or len(gender_accuracy["female"]) == 0:
        return
    
    plt.figure()
    plt.plot(gender_accuracy['male'])
    plt.plot(gender_accuracy['female'])
    plt.title('Gender Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Male', 'Female'], loc='upper left')

    plt.show()

def plot_confusion_matrix(results, benchmark, dataset_name):
    '''
    Plot the confusion matrix of the training results, representing the epoch with best validation accuracy
    Arguments:
        results (dictionary) - containing keys 'y_true' and 'y_pred' with value of type List 
    '''
    cm = confusion_matrix(results['y_true'], results['y_pred'])

    plt.figure(figsize=(10, 8))
    
    if dataset_name == 'FER2013':
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=FER_CLASS_LABELS, yticklabels=FER_CLASS_LABELS)
    else:
        daisee_labels = [x + "-" + benchmark for x in DAISEE_CLASS_LABELS_DEGREES]
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=daisee_labels, yticklabels=daisee_labels)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_predictions(model, loader, benchmark, dataset_name):
    '''
    Plot model predictions with true and predicted labels of the DataLoader 'loader'
    Arguments:
        model - a PyTorch model
        loader (iterable) - a DataLoader class from PyTorch which loads the Dataset
        benchmark (string) - Defines the benchmark metric to reference in labelling of figures. Must be one of "Boredom", "Engagement", "Confusion", "Frustration"
        dataset_name (string) - Defines the class label to reference in labelling of figures. Must be one of "FER2013" or "DAiSEE" 
    '''
    model.to(DEVICE)
    model.eval()
    _, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})

    # Make prediction on test set
    with torch.no_grad():
        inputs, labels, _ = next(iter(loader))
        
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if benchmark:
            labels = loops.benchmark_to_labels(labels, benchmark).to(DEVICE)
        
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs, 1)

        for i, ax in enumerate(axes.flat):
            image = inputs[i].cpu().numpy().transpose(1, 2, 0)
            true_label = int(labels[i].cpu())
            pred_label = int(predicted[i].cpu())
            
            ax.imshow(image, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            
            if dataset_name == "FER2013":
                ax.set_title(f'True: {FER_CLASS_MAP[true_label]}\nPredicted: {FER_CLASS_MAP[pred_label]}', color=color)
            else:
                ax.set_title(f'True: {DAISEE_CLASS_LABELS_DEGREES[true_label]}-{benchmark}\nPredicted: {DAISEE_CLASS_LABELS_DEGREES[pred_label]}-{benchmark}', color=color)
                
    plt.tight_layout()
    plt.show()
    
def plot_image_features(image_path, bbox, landmarks):
    '''
    Plot face bbox (red) and landmarks (green) on image_path in grayscale
    '''
    landmark_labels = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, ax = plt.subplots()
    
    for i, point in enumerate(landmarks):
        x, y = point
        circle = patches.Circle((x, y), radius=1, color='lime')
        ax.add_patch(circle)
        cv2.putText(image, landmark_labels[i], (x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255), thickness=2)
        
    ax.imshow(image, cmap='gray')
    rect = detect.bbox_to_rect(bbox)
    ax.add_patch(rect)
    
def display_classification_report(results, benchmark, dataset_name):
    '''
    Display the classification report of the training results, representing the epoch with best validation accuracy
    Arguments:
        results (dictionary) - containing keys 'y_true' and 'y_pred' with value of type List 
        dataset_string (string) - Defines the reference dataset. Must be one of "FER2013" or "DAiSEE"
    '''
    print('=== Classification Report ===')
    if dataset_name == 'FER2013':
        print(classification_report(results['y_true'], results['y_pred'], target_names=FER_CLASS_LABELS, digits=4))
    else:
        daisee_labels = [x + "-" + benchmark for x in DAISEE_CLASS_LABELS_DEGREES]        
        print(classification_report(results['y_true'], results['y_pred'], target_names=daisee_labels, digits=4))

def plot_compare_predictions(model1, model2, loader, benchmark, dataset_name):
    '''
    Plot predictions of two models with true and predicted labels of DataLoader 'loader'
    Arguments:
        model1 - a PyTorch model
        model2 - a PyTorch model
        loader (iterable) - a DataLoader class from PyTorch which loads the Dataset
        benchmark (string) - Defines the benchmark metric to reference in labelling of figures. Must be one of "Boredom", "Engagement", "Confusion", "Frustration"
        dataset_name (string) - Defines the class label to reference in labelling of figures. Must be one of "FER2013" or "DAiSEE" 
    '''
    model1.to(DEVICE)
    model1.eval()
    model2.to(DEVICE)
    model2.eval()
    _, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10), subplot_kw={'xticks': [], 'yticks': []})
    
    # Make prediction
    with torch.no_grad():
        inputs, labels, _ = next(iter(loader))
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
        if benchmark:
            labels = loops.benchmark_to_labels(labels, benchmark).to(DEVICE)
            
        outputs1 = model1(inputs.float())
        outputs2 = model2(inputs.float())
        _, predicted1 = torch.max(outputs1, 1)
        _, predicted2 = torch.max(outputs2, 1)

        for i, ax in enumerate(axes.flat):
            image = inputs[i].cpu().numpy().transpose(1, 2, 0)
            true_label = int(labels[i].cpu())
            pred_label1 = int(predicted1[i].cpu())
            pred_label2 = int(predicted2[i].cpu())
            ax.imshow(image, cmap='gray')
            
            # Green indicates model2 improvement over model1
            color = 'green' if true_label == pred_label2 and true_label != pred_label1 else 'red'
            
            if dataset_name == "FER2013":
                ax.set_title(f'True: {FER_CLASS_MAP[true_label]}\nPredicted (simple): {FER_CLASS_MAP[pred_label1]} {"O" if true_label == pred_label1 else "X"}\nPredicted (occlusion_aware): {FER_CLASS_MAP[pred_label2]} {"O" if true_label == pred_label2 else "X"}', color=color)
            else:
                ax.set_title(f'True: {DAISEE_CLASS_LABELS_DEGREES[true_label]}-{benchmark}\nPredicted (simple): {DAISEE_CLASS_LABELS_DEGREES[pred_label1]}-{benchmark} {"O" if true_label == pred_label1 else "X"}\nPredicted (occlusion_aware): {DAISEE_CLASS_LABELS_DEGREES[pred_label2]}-{benchmark} {"O" if true_label == pred_label2 else "X"}', color=color)
            
    plt.tight_layout()
    plt.show()
