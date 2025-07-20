import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np



class CelebADataset(Dataset):
    def __init__(self, img_path='../data/CelebA', img_size=64):
        """
        Initialize the CelebADataset class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. Defaults to '../data/CelebA'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        """
        # Define data transformations
        transform = transforms.Compose([transforms.Resize((img_size, img_size)),  # Resize images to 64x64
                                        transforms.ToTensor()]) # Convert images to tensors
        self.image_dataset = torchvision.datasets.ImageFolder(root=img_path, transform=transform)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        return self.image_dataset[idx]
    
class CelebAData():
    def __init__(self, img_path='../../data/CelebA/dataset', img_size=64, max_size=None):
        """
        Initialize the CelebAData class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. 
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        max_size : int, optional
            Maximum size of the dataset. If None, use the full dataset. Defaults to None.
        """
        self.dataset = CelebADataset(img_path, img_size)
        if max_size is not None:
            self.dataset = Subset(self.dataset, list(range(max_size)))

    def get_dataloader(self, batch_size=128, shuffle=True, drop_last=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class CelebAAttrDataset(Dataset):
    def __init__(self, img_path='../data/CelebA', attr_path='../data/CelebA/list_attr_celeba.csv', attr_name='Eyeglasses', img_size=64):
        # Define data transformations
        """
        Initialize the CelebAAttrDataset class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. Defaults to '../data/CelebA'.
        attr_path : str, optional
            Path to the CSV containing the attribute labels. Defaults to '../data/CelebA/list_attr_celeba.csv'.
        attr_name : str, optional
            Name of the attribute to use as the label. Defaults to 'Eyeglasses'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        """
        transform = transforms.Compose([transforms.Resize((img_size, img_size)),  # Resize images to 64x64
                                        transforms.ToTensor()]) # Convert images to tensors
        self.image_dataset = torchvision.datasets.ImageFolder(root=img_path, transform=transform)
        df = pd.read_csv(attr_path)
        self.labels = torch.tensor(df[attr_name].values, dtype=torch.float32)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]  # Ignore the folder class label
        label = self.labels[idx]
        return image, label
    

class CelebAAttrData():
    def __init__(self, img_path='../../data/CelebA/dataset', attr_path='../../data/CelebA/dataset/list_attr_celeba.csv', attr_name='Eyeglasses', img_size=64, max_size=None):
        """
        Initialize the CelebAAttrData class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. 
        attr_path : str, optional
            Path to the CSV containing the attribute labels. 
        attr_name : str, optional
            Name of the attribute to use as the label. Defaults to 'Eyeglasses'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        max_size : int, optional
            Maximum size of the dataset. If None, use the full dataset. Defaults to None.
        """
        self.dataset = CelebAAttrDataset(img_path, attr_path, attr_name, img_size)
        if max_size is not None:
            self.dataset = Subset(self.dataset, list(range(max_size)))
    
    def get_dataloader(self, batch_size=128, shuffle=True, drop_last=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    

def convert_attr_txt2csv(input_txt='../../data/CelebA/list_attr_celeba.txt', output_csv='../../data/CelebA/list_attr_celeba.csv'):
    """
    Convert the CelebA attribute text file to a CSV file.

    Parameters
    ----------
    input_txt : str, optional
        Path to the input text file. Defaults to '../../data/CelebA/list_attr_celeba.txt'.
    output_csv : str, optional
        Path to the output CSV file. Defaults to '../../data/CelebA/list_attr_celeba.csv'.

    Notes
    -----
    The text file is downloaded from the CelebA dataset website.
    The first row of the text file is skipped, as it contains column names.
    The -1/1 values are converted to 0/1, which is useful for binary classification.
    The index column is converted from '000001.jpg' to integer 1, etc.
    The dataframe is saved to CSV with the index column included.
    """
    df = pd.read_csv(input_txt, delim_whitespace=True, skiprows=1)
    # Convert -1/1 values to 0/1 (optional, useful for binary classification)
    df = df.replace(-1, 0)
    # Convert the index (filename column) from '000001.jpg' to integer 1, etc.
    df.index = df.index.str.replace(".jpg", "").astype(int)
    df.index.name = "index"
    # Save the dataframe to CSV
    df.to_csv(output_csv, index=True)
    print(f"Attributes successfully saved to {output_csv}")


class CelebAAttrSameDataset(Dataset):
    def __init__(self, img_path='../../data/CelebA', attr_path='../../data/CelebA/list_attr_celeba.csv', attr_name='Eyeglasses', img_size=64, flag=1):
        # Define data transformations
        """
        Initialize the CelebAAttrSameDataset class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. Defaults to '../../data/CelebA'.
        attr_path : str, optional
            Path to the CSV containing the attribute labels. Defaults to '../../data/CelebA/list_attr_celeba.csv'.
        attr_name : str, optional
            Name of the attribute to filter the images by. Defaults to 'Eyeglasses'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        flag : int, optional
            Value of the attribute used to filter images. Defaults to 1.

        Notes
        -----
        The dataset will only include images where the specified attribute matches the given flag.
        """

        transform = transforms.Compose([transforms.Resize((img_size, img_size)),  # Resize images to 64x64
                                        transforms.ToTensor()]) # Convert images to tensors
        self.image_dataset = torchvision.datasets.ImageFolder(root=img_path, transform=transform)
        df = pd.read_csv(attr_path)
        indices = np.where(df[attr_name].values == flag)[0].tolist()
        self.image_dataset = Subset(self.image_dataset, indices)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        return self.image_dataset[idx]



class CelebAAttrSameData():
    def __init__(self, img_path='../data/CelebA', attr_path='../data/CelebA/list_attr_celeba.csv', attr_name='Eyeglasses', img_size=64, flag=1):
        """
        Initialize the CelebAAttrSameDataLoader class.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. Defaults to '../data/CelebA'.
        attr_path : str, optional
            Path to the CSV containing the attribute labels. Defaults to '../data/CelebA/list_attr_celeba.csv'.
        attr_name : str, optional
            Name of the attribute to filter the images by. Defaults to 'Eyeglasses'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        flag : int, optional
            Value of the attribute used to filter images. Defaults to 1.

        Notes
        -----
        The dataset will only include images where the specified attribute matches the given flag.
        """
        self.dataset = CelebAAttrSameDataset(img_path, attr_path, attr_name, img_size, flag)
    
    def get_dataloader(self, batch_size=128, shuffle=True, drop_last=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        


class CelebAAttrFRData():
    def __init__(self, img_path='../../data/CelebA/dataset', attr_path='../../data/CelebA/dataset/list_attr_celeba.csv', attr_name='Eyeglasses', img_size=64, max_size=None):
        """
        The CelebAAttrFRData class produces forget and retain dataloaders for a specific attribute.

        Parameters
        ----------
        img_path : str, optional
            Path to the folder containing the CelebA images. 
        attr_path : str, optional
            Path to the CSV containing the attribute labels. 
        attr_name : str, optional
            Name of the attribute to use as the label. Defaults to 'Eyeglasses'.
        img_size : int, optional
            Size of the images to resize to. Defaults to 64.
        max_size : int, optional
            Maximum size of the dataset. If None, use the full dataset. Defaults to None.
        """
        self.dataset_f = CelebAAttrSameDataset(img_path, attr_path, attr_name, img_size, flag=1)
        self.dataset_r = CelebAAttrSameDataset(img_path, attr_path, attr_name, img_size, flag=0)
        if max_size is not None:
            self.dataset_f = Subset(self.dataset_f, list(range(max_size)))
            self.dataset_r = Subset(self.dataset_r, list(range(max_size)))

    def get_dataloader_rf(self, batch_size=128, shuffle=True, drop_last=True):
        dl_f = DataLoader(self.dataset_f, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        dl_r = DataLoader(self.dataset_r, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dl_r, dl_f
