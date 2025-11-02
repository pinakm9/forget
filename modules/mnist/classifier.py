import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import sys, os
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np


# Add the 'modules' directory to the Python search path
sys.path.append(os.path.abspath('../modules'))
import utility as ut
import datapipe



# Define a simple classifier model.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        # self.bn1   = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):
        # Flatten the input tensor (N, 1, 28, 28) to (N, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x



class EfficientMNISTClassifier(nn.Module):
    def __init__(self):
        super(EfficientMNISTClassifier, self).__init__()
        # Convolutional layers with Batch Normalization.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # (32, 28, 28)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # (64, 28, 28)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # (128, 28, 28)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.pool  = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions to 14x14.
        
        # 1x1 convolution to expand channels.
        self.conv4 = nn.Conv2d(128, 200, kernel_size=1)             # (200, 14, 14)
        self.bn4   = nn.BatchNorm2d(200)
        
        # Fully connected layers.
        # After global average pooling, we have 200 features.
        self.fc1 = nn.Linear(200, 148)
        self.dropout = nn.Dropout(0.3)  # Moderate dropout to improve generalization.
        self.fc2 = nn.Linear(148, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 28, 28)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, 28, 28)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 128, 28, 28)
        x = self.pool(x)                     # (batch, 128, 14, 14)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch, 200, 14, 14)
        x = F.adaptive_avg_pool2d(x, (1, 1))   # Global average pooling → (batch, 200, 1, 1)
        x = x.view(x.size(0), -1)              # Flatten to (batch, 200)
        x = F.relu(self.fc1(x))                # (batch, 148)
        x = self.dropout(x)                    # Apply dropout
        x = self.fc2(x)                        # (batch, 10)
        return x
    





def train(folder, architecture, num_epochs, batch_size, all_digits=list(range(10)), data_path="../../data/MNIST", gen_data=False, gen_model=None):
    # Create DataLoaders for the training and test sets.
    train_loader = datapipe.MNIST().get_dataloader(root=data_path,batch_size=batch_size, train=True, all_digits=all_digits)
    test_loader  = datapipe.MNIST().get_dataloader(root=data_path, batch_size=batch_size, train=False, all_digits=all_digits)

    ut.makedirs(folder)
    
    # Instantiate the model.
    model = architecture()

    # Move the model to GPU (or MPS) if available.
    device = ut.get_device()
    model.to(device)
    model.train()

    if gen_data:
        gen_model.to(device)
        gen_model.eval()

    # Define the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Wrap the epoch loop with tqdm.

    epoch_iter = tqdm.tqdm(range(num_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            if gen_data:
                data, _, _ = gen_model(data.view(data.shape[0], -1))
                data = data.reshape(data.shape[0], 1, 28, 28)
            
            # print(data.shape, target.shape)
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
                if gen_data:
                    data, _, _ = gen_model(data.view(data.shape[0], -1))
                    data = data.reshape(data.shape[0], 1, 28, 28)
                output = model(data)
                # Get predictions from the maximum value.
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_acc = 100.0 * correct / total

        # Update the tqdm progress bar with the latest metrics.
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", test_acc=f"{test_acc:.2f}%")
    
    # Save the final model's state_dict.
    torch.save(model, f"{folder}/{architecture.__name__}.pth")


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
        model = torch.load(model)
    model.to(device)
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
    model = torch.load(path, weights_only=False, map_location=device)
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
    return torch.bincount(torch.argmax(logits, dim=1), minlength=10)


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
    # combined_score = weight_entropy * entropy_scores + weight_margin * margin_scores
    return entropy_scores, margin_scores#, combined_score


def compute_features(images, identifier):
    """
    Extract features from a list of images using the identifier's intermediate layer.
    The identifier is expected to have a method `get_features` that returns the penultimate features.
    """
    images = images.view(images.size(0), -1)
    features = torch.relu(F.linear(images.cpu(), identifier.fc1.weight.cpu(), identifier.fc1.bias.cpu()))
    return features.detach().cpu().numpy() if images.device.type != "cuda" else features.detach().numpy()


def frechet_inception_distance(real_images, gen_images, identifier):
    """
    Computes the Frechet Inception Distance (FID) between two sets of images using NumPy.
    
    FID is computed as the squared difference between the mean feature vectors of the two sets,
    plus the trace of the sum of the covariance matrices minus twice the matrix square root 
    of their product.
    
    Parameters
    ----------
    real_images : array-like
        Real images.
    gen_images : array-like
        gen images.
    identifier : callable
        A function or network that computes features from images. It should accept an image (or batch)
        and return a NumPy array of features.
        
    Returns
    -------
    fid : float
        The FID between the real and gen images.
    """
    # Extract features from real and gen images
    real_features = compute_features(real_images, identifier)  # shape: (N, D)
    gen_features = compute_features(gen_images, identifier) # shape: (M, D)
    
    

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    cov_sqrt = sqrtm(sigma_real @ sigma_gen)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Compute FID score
    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fid

#

def inception_score(logits):
    preds = F.softmax(logits, dim=1)
    
    # Compute the marginal distribution p(y)
    py = torch.mean(preds, axis=0)
    
    # Compute the KL divergence for each image and average
    kl_divs = torch.sum(preds * (torch.log(preds + 1e-9) - torch.log(py + 1e-9)), dim=1)
    return torch.exp(kl_divs.mean())


