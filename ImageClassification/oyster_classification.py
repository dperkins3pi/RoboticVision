import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import ResNet50_Weights
from matplotlib import pyplot as plt
from matplotlib import cm
import torch.optim as optim

#############################Confusion Matrix#############################

def get_conf_mat(y_pred, y_target, n_cats):
    """Build confusion matrix from scratch.
    (This part could be a good student assignment.)
    """
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm

def vis_conf_mat(conf_mat, cat_names, acc):
    """Visualize the confusion matrix and save the figure to disk."""
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
            ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    # plt.show()
    _filename = 'conf_mat_oyster.png'
    plt.savefig(_filename, bbox_inches='tight')


#############################Load In The Data#############################
data_dir = "Oyster Shell"
train_dir = f"{data_dir}/train"
test_dir = f"{data_dir}/test"

# Augmentations for training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally with 50% chance
    transforms.RandomRotation(15),  # Rotate randomly within 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color changes
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations for test set (no augmentation, just resizing and normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Each element of the lader is an image (batch_size x 3 x 224 x 224) and a label (batch_size)

print("The data has been loaded. The classes are labeled as:")
print(train_dataset.class_to_idx)


#############################Train A ResNet#############################

print("\nTraining the Model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained ResNet
num_classes = len(train_dataset.class_to_idx.values())
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Make final output be the correct dimension
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer
epochs = 10
max_acc = 0

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_training_loss = 0.0
    correct_train = 0
    total_train = 0
    running_eval_loss = 0.0
    correct_eval = 0
    total_eval = 0
    
    print(f"Epoch {epoch+1}/{epochs}") 
    
    # Train it on training loader
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward and backward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_training_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_accuracy = 100 * correct_train / total_train
    print(f"\tTrain_Loss: {running_training_loss/len(train_loader)},  Train_Acc: {train_accuracy:.2f}%")
    
    # Gather metric on the test_loader
    model.eval() 
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_eval_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_eval += labels.size(0)
        correct_eval += (predicted == labels).sum().item()
        
    eval_accuracy = 100 * correct_eval / total_eval
    if eval_accuracy > max_acc:
        max_acc = eval_accuracy
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, 'best_model_oyster.pth')
        
    print(f"\tEval_Loss: {running_eval_loss/len(test_loader)},  Eval_Acc: {eval_accuracy:.2f}%")
        

#############################Get The Results#############################

print("\nGetting the confusion matrix")

# Download the best model
model = models.resnet50(weights=None)  # No pre-trained weights here
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Make final output be the correct dimension
model.load_state_dict(torch.load('best_model_oyster.pth'))

model.eval()  # Set to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Forward pass through the model and get the predicted class
        inputs = inputs.to(device) 
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # Append the predictions and labels to the list
        y_pred.extend(predicted.cpu().numpy())
        labels = labels.to(device)
        y_true.extend(labels.cpu().numpy())  # Store the true labels
        
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Output the confusion matrix
cmat = get_conf_mat(y_pred=y_pred, y_target=y_true, n_cats=num_classes)
acc = cmat.trace() / cmat.shape[0]
vis_conf_mat(cmat, list(train_dataset.class_to_idx.keys()), acc)

# ecen_venv\Scripts\activate