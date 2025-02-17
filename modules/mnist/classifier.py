import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import sys, os
import torch.nn.functional as F


# Add the 'modules' directory to the Python search path
sys.path.append(os.path.abspath('../modules'))
import utility as ut
import datapipe



# Define a simple classifier model.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # Flatten the input tensor (N, 1, 28, 28) to (N, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTClassifierConv(nn.Module):
    def __init__(self):
        super(MNISTClassifierConv, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No Softmax because CrossEntropyLoss handles it
        return x


def train(folder, num_epochs, batch_size):
    # Create DataLoaders for the training and test sets.
    train_loader = datapipe.MNIST().get_dataloader(batch_size=batch_size, train=True)
    test_loader  = datapipe.MNIST().get_dataloader(batch_size=batch_size, train=False)

    checkpoint_dir = f"{folder}/checkpoints"
    ut.makedirs(checkpoint_dir)
    
    # Instantiate the model.
    model = MNISTClassifier()

    # Move the model to GPU (or MPS) if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)

    # Define the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Wrap the epoch loop with tqdm.
    epoch_iter = tqdm.tqdm(range(num_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()       # Zero the gradients.
            output = model(data)        # Forward pass.
            loss = criterion(output, target)
            loss.backward()             # Backward pass.
            optimizer.step()            # Update parameters.
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluate on the test set.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Get predictions from the maximum value.
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_acc = 100.0 * correct / total

        # Update the tqdm progress bar with the latest metrics.
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", test_acc=f"{test_acc:.2f}%")
    
    # Save the final model's state_dict.
    torch.save(model.state_dict(), f"{checkpoint_dir}/classifier.pth")


# @ut.timer
def count_digit(images, digit, model="../data/MNIST/classifier/checkpoints/classifier.pth", device="cuda"):
    """
    Count the number of images classified as a specific digit using the provided model.
    
    Parameters:
        model: A trained digit classifier (e.g., an instance of MNISTClassifier).
               The model should accept inputs of shape (N, 1, 28, 28) and output logits of shape (N, 10).
        images (torch.Tensor): A tensor containing digit images with shape (N, 1, 28, 28).
        digit (int): The digit to count (e.g., 1 for counting ones).
        device (str): The device on which to run the model ('cuda' or 'cpu').
        
    Returns:
        int: The number of images that are classified as the specified digit.
    """
    if isinstance(model, str):
        state_dict = torch.load(model)
        model = MNISTClassifier().to(device)
        model.load_state_dict(state_dict)
    # Set the model to evaluation mode.
    model.eval()
    # Move the images to the specified device.
    images = images.to(device)

    # Disable gradient calculation for inference.
    with torch.no_grad():
        # try:
        #     logits = model(images.reshape(-1, 1, 28, 28))
        # except:
        logits = model(images)
              # Get the raw logits from the model.
        # Determine predicted classes by taking the argmax over the logits.
        predictions = torch.argmax(logits, dim=1)
    
    # Count the number of predictions that equal the specified digit.
    count = (predictions == digit).sum().item()
    return count


def get_classifier(path="../data/MNIST/classifier/checkpoints/classifier.pth", device="cuda"):
    """
    Load a pre-trained digit classifier from a checkpoint file and return it as an instance of MNISTClassifier.
    
    Parameters:
        folder (str): The folder containing the checkpoint file (not used).
        path (str): The path to the checkpoint file (default: '../data/MNIST/classifier/checkpoints/classifier.pth').
        device (str): The device on which to run the model ('cuda' or 'cpu').
        
    Returns:
        MNISTClassifier: An instance of MNISTClassifier with the pre-trained weights.
    """
    model = MNISTClassifier()
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

def count_from_logits(logits):
    """
    Count the number of occurrences of each digit in the input logits.

    Parameters:
        logits (torch.Tensor): The input logits, of shape (batch_size, num_classes).

    Returns:
        tuple: A tuple of two elements:
            - The bincount of the argmax of the logits, of shape (num_classes,).
            - The top 2 elements of the softmax of the logits, of shape (batch_size, 2).
    """
    return torch.bincount(torch.argmax(logits, dim=1))


def entropy(probs):
    """Computes entropy-based uncertainty from probabilities."""
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Avoid log(0)

def margin_confidence(probs):
    """Computes margin-based uncertainty (inverted: lower margin means more ambiguous) from probabilities."""
    top2_probs, _ = torch.topk(probs, 2, dim=-1)  # Get top 2 probabilities
    margin = top2_probs[:, 0] - top2_probs[:, 1]  # Difference between top two
    return 1 - margin  # Invert: low margin = high ambiguity

def ambiguity(logits, weight_entropy=0.5, weight_margin=0.5):
    """Combines entropy and margin-based ambiguity detection."""
    probs = torch.softmax(logits, dim=-1) 
    entropy_scores = entropy(probs)
    margin_scores = margin_confidence(probs)
    
    # Normalize scores to [0, 1] (optional for better scaling)
    entropy_scores = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-9)
    margin_scores = (margin_scores - margin_scores.min()) / (margin_scores.max() - margin_scores.min() + 1e-9)
    
    # Weighted combination
    combined_score = weight_entropy * entropy_scores + weight_margin * margin_scores
    return entropy_scores, margin_scores, combined_score
