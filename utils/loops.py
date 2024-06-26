import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0.0, 0.0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        valid_accuracy = correct_predictions / total_samples    
        valid_loss = running_loss / total_samples

        print(f'validation_loss: {valid_loss:.4f} - valid_accuracy: {valid_accuracy:.4f}')
            
    return valid_accuracy, valid_loss, y_true, y_pred

def train_model(model, train_loader, optimizer, criterion, scaler, epoch, num_epochs):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0.0, 0.0
    loop = tqdm(train_loader, leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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