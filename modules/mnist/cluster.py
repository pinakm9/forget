import torch 
import matplotlib.pyplot as plt
import numpy as np
import classifier as cl


def extract_features(model, img, device='mps'):
    with torch.no_grad():
        return model.encoder(img.view(img.shape[0], -1).to(device)).cpu().numpy() 
    

def extract_latent(model, img, device='mps'):
    with torch.no_grad():
        return model.encode(img.view(img.shape[0], -1).to(device))[0].cpu().numpy()
    
def extract_recon_latent(model, noise, identifier, device='mps'):
    # z_random = torch.randn(num, 2).to(device)
    img = model.decoder(noise)
    with torch.no_grad():
        features = model.encode(img.view(img.shape[0], -1).to(device))[0]
    logits = identifier(img)
    return features.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy()


def kmeans(X, num_clusters, num_iters=100, device='mps'):
    # Randomly initialize centroids
    X = torch.tensor(X, device=device)
    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices]

    for _ in range(num_iters):
        # Compute distances between points and centroids
        distances = torch.cdist(X, centroids)  # (N, K)
        labels = distances.argmin(dim=1)

        # Update centroids
        for k in range(num_clusters):
            if (labels == k).sum() > 0:
                centroids[k] = X[labels == k].mean(dim=0)

    return labels.cpu().numpy(), centroids.cpu().numpy()

def generate_continuous_colors(n, cmap_name='viridis'):
    """
    Generate n visually distinct continuous colors using a colormap.
    
    Args:
        n (int): Number of colors to generate.
        cmap_name (str): Name of the matplotlib colormap (e.g., 'viridis', 'plasma', 'coolwarm').
    
    Returns:
        List of RGB tuples (each in range 0–1).
    """
    cmap = plt.get_cmap(cmap_name)
    return np.array([cmap(i / (n - 1)) for i in range(n)])


def cluster_dist(cluster_labels, real_labels):
    m = len(np.unique(cluster_labels))
    n = 10 #len(np.unique(real_labels))
    z = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            idx = np.where(cluster_labels == i)[0]
            z[i, j] = (real_labels[idx] == j).sum() / len(idx)
    return z