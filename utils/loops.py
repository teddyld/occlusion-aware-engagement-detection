import torch
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import utils.fer2013 as fer2013
import utils.daisee as daisee

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmark_to_labels(labels, benchmark):
    """
    Convert one hot encoded labels to single label for benchmark metric
    """
    if benchmark == "Boredom":
        labels = labels[:, :4]
    elif benchmark == "Engagement":
        labels = labels[:, 4:8]
    elif benchmark == "Confusion":
        labels = labels[:, 8:12]
    elif benchmark == "Frustration":
        labels = labels[:, 12:]
    else:
        raise ValueError('Invalid benchmark metric')
    
    targets = torch.argmax(labels, dim=1)
    return targets

def get_train_class_weights(dataset_name, benchmark, DEVICE=None):
    '''
    Return class weights of train dataset
    '''
    if not DEVICE:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == 'FER2013':
        train_dataset, _, _ = fer2013.get_datasets()
    else: 
        train_dataset, _, _ = daisee.get_datasets()
    labels = train_dataset.get_labels()
    
    if benchmark:    
        labels = benchmark_to_labels(torch.tensor(labels), benchmark).tolist()
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights).float().to(DEVICE)

def evaluate_model(model, loader, criterion, benchmark):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0.0, 0.0
    y_pred = []
    y_true = []
    correct_male_predictions, correct_female_predictions = 0.0, 0.0
    total_male_samples, total_female_samples = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels, genders in loader:
            inputs, labels, genders = inputs.to(DEVICE), labels.to(DEVICE), genders.to(DEVICE)
            if benchmark:
                labels = benchmark_to_labels(labels, benchmark).to(DEVICE)
            
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            correct_male_predictions += (predicted == labels).where(genders==0, False).sum().item()
            correct_female_predictions += (predicted == labels).where(genders==1, False).sum().item()
            
            total_samples += labels.size(0)
            total_male_samples += genders.eq(0).sum().item()
            total_female_samples += genders.eq(1).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
        valid_accuracy = correct_predictions / total_samples    
        valid_loss = running_loss / total_samples
        male_accuracy = correct_male_predictions / total_male_samples
        female_accuracy = correct_female_predictions / total_female_samples

        print(f'validation_loss: {valid_loss:.4f} - valid_accuracy: {valid_accuracy:.4f}')
        if benchmark:
            print(f'male_accuracy: {male_accuracy:.4f} - female_accuracy: {female_accuracy:.4f}')
            
    return valid_accuracy, valid_loss, y_true, y_pred, male_accuracy, female_accuracy

def train_model(model, train_loader, optimizer, criterion, scaler, epoch, num_epochs, benchmark):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0.0, 0.0
    loop = tqdm(train_loader, leave=False)
    for inputs, labels, _ in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        if benchmark:
            labels = benchmark_to_labels(labels, benchmark).to(DEVICE)

        optimizer.zero_grad()
        
        # Forward + Backward + Optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        
    train_accuracy = correct_predictions / total_samples    
    train_loss = running_loss / total_samples
    
    return train_accuracy, train_loss