from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class MNIST:
    def __init__(self):
        """
        Initialize the MNIST dataset.
        """
        super(MNIST, self).__init__()


    def get_dataloader(self, batch_size, pad=False, root='../data', train=True, all_digits=list(range(10))):
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
        if len(all_digits) < 10:
            dataset = self.filter_mnist(dataset, all_digits)

        return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)



    def get_dataloader_rf(self, batch_size, pad=False, root='../data', train=True, all_digits=list(range(10)), forget_digit=1):    
        """
        Get DataLoaders for an MNIST dataset that splits the data into two groups: 
        one containing only the 'forget' digit and the other containing the remaining digits.

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
        all_digits : list of int, optional
            The list of digit classes to consider. Default is all digits [0-9].
        forget_digit : int, optional
            The digit to separate from the rest. Default is 1.

        Returns
        -------
        tuple of DataLoader
            A tuple containing two DataLoaders: the first for the 'forget' digit subset,
            and the second for the 'retain' digits subset.
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
        if len(all_digits) < 10:    
            dataset = self.filter_mnist(dataset, all_digits)
            
        # Get indices of the forget_digit and the retain digits
        retain_indices = [i for i, (_, label) in enumerate(dataset) if label != forget_digit]
        forget_indices = [i for i, (_, label) in enumerate(dataset) if label == forget_digit]
        
        # Create subset for the forget_digit and the retain digits
        retain_subset = Subset(dataset, retain_indices)
        forget_subset = Subset(dataset, forget_indices)
        
        # Create DataLoaders
        dataloader_retain = DataLoader(retain_subset, batch_size=batch_size, shuffle=True)
        dataloader_forget = DataLoader(forget_subset, batch_size=batch_size, shuffle=True)
        return  dataloader_retain, dataloader_forget


    def filter_mnist(self, dataset, all_digits=[1, 3, 8]):
        """
        Filter the MNIST dataset to include only the specified digits.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The MNIST dataset to filter.
        all_digits : list of int, optional
            The list of digit classes to include in the filtered dataset. Default is [1, 3, 8].

        Returns
        -------
        torch.utils.data.Subset
            A subset of the original dataset containing only the specified digits.
        """

        indices = [i for i, (img, label) in enumerate(dataset) if label in all_digits]
        filtered_dataset = Subset(dataset, indices)
        return filtered_dataset




   
    

    