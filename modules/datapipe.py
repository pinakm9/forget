from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class MNIST:
    def __init__(self):
        """
        Initialize the MNIST dataset.
        """
        super(MNIST, self).__init__()


    def get_dataloader(self, batch_size, pad=False, root='../data', train=True):
        """
        Get a dataloader for the MNIST dataset.

        Parameters
        ----------
        batch_size : int
            The batch size.
        pad : bool, optional
            Whether to pad the images with zeros. Default is False.
        root : str, optional
            The root directory of the dataset. Default is '../data'.
        train : bool, optional
            Whether to get the training set or the test set. Default is True.

        Returns
        -------
        DataLoader
            A DataLoader for the MNIST dataset.
        """
        if pad:
            transform = transforms.Compose([
                    transforms.Pad(padding = 2, padding_mode = 'edge'),
                    transforms.ToTensor()
                    ])
        else:
            transform = transforms.ToTensor()
        
        dataset = datasets.MNIST(root = root, train = train, download = True, 
                        transform = transform)
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size, 
                                shuffle = True)
        return dataloader



    def get_dataloader_one_vs_rest(self, batch_size, pad=False, root='../data', train=True):
        """
        Get two DataLoaders for the MNIST dataset, one for digit '1' and the other for the rest.

        Parameters
        ----------
        batch_size : int
            The batch size.
        pad : bool, optional
            Whether to pad the images with zeros. Default is False.
        root : str, optional
            The root directory of the dataset. Default is '../data'.
        train : bool, optional
            Whether to get the training set or the test set. Default is True.

        Returns
        -------
        tuple of two DataLoaders
            Two DataLoaders, the first one containing only digit '1' and the second one containing the rest.
        """
        if pad:
            transform = transforms.Compose([
                    transforms.Pad(padding = 2, padding_mode = 'edge'),
                    transforms.ToTensor()
                    ])
        else:
            transform = transforms.ToTensor()
        
        dataset = datasets.MNIST(root = root, train = train, download = True, 
                        transform = transform)
        # Get indices of digit '1' and the rest
        one_indices = [i for i, label in enumerate(dataset.targets) if label == 1]
        rest_indices = [i for i, label in enumerate(dataset.targets) if label != 1]
        # Create subset for digit '1' and the rest
        one_subset = Subset(dataset, one_indices)
        rest_subset = Subset(dataset, rest_indices)
        # Create DataLoaders
        dataloader_one = DataLoader(one_subset, batch_size=batch_size, shuffle=True)
        dataloader_rest = DataLoader(rest_subset, batch_size = batch_size, shuffle = True)
        return dataloader_one, dataloader_rest


    


