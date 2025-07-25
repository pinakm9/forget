import pandas as pd
import numpy as np
import json
import utility as ut
import vae_viz as viz
import os, glob
import matplotlib.pyplot as plt
from vae import VAE 
import torch
from torch.linalg import eigh, svd
import classifier as cl
import datapipe, csv
from torchvision.utils import save_image
from fid_inceptionv3 import compute_inception_features
import vae_train as vt



class BatchExperiment:
    def __init__(self, train_func, train_kwargs, n_exprs):
        """
        Constructor for BatchExperiment.

        Parameters
        ----------
        train_func : function
            The function to be used for training.
        train_kwargs : dict
            The keyword arguments to be passed to the training function.
        n_exprs : int
            The number of experiments to run.
        """

        self.train_func = train_func
        self.train_kwargs = train_kwargs
        self.n_exprs = n_exprs  

    @ut.timer
    def run(self):
        """
        Run the training loop in parallel over n_exprs experiments.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in range(self.n_exprs):
            train_kwargs = self.train_kwargs.copy()
            train_kwargs['folder'] = self.get_folder(i)
            self.train_func(**train_kwargs)
        # self.summarize()

    def get_folder(self, index):
        """
        Generate the folder path for a specific experiment index.

        Parameters
        ----------
        index : int
            The index of the experiment for which to generate the folder path.

        Returns
        -------
        str
            The folder path for the specified experiment index.
        """

        return f'{self.train_kwargs["folder"]}/expr-{index}'

    def summarize(self, threhold=2e-2, num_fid_samples=25000, device='cuda', batch_size=256):
        """
        Summarize the training curves of all experiments and save the statistics to summary.json and summary_std.json

        Parameters
        ----------
        threhold : float, optional
            The threshold to find the stable stopping point, by default 2e-2
        num_fid_samples : int, optional 
            The number of samples to use for FID computation, by default 25000
        device : str or torch.device, optional
            The device to use for computation, by default 'cuda'
        batch_size : int, optional
            The batch size to use for FID computation, by default 256

        Returns
        -------
        None
        """
        if not os.path.exists(f'{self.train_kwargs["folder"]}/checkpoints'):
            os.makedirs(f'{self.train_kwargs["folder"]}/checkpoints')

        mean, std, summary, summary_std = {}, {}, {}, {}
        df0 = pd.read_csv(f'{self.get_folder(0)}/checkpoints/training_log.csv')
        arr = df0.to_numpy()
        data = np.zeros((self.n_exprs, arr.shape[0], arr.shape[1]))
        for i in range(self.n_exprs):
            folder = self.get_folder(i)
            data[i] = pd.read_csv(f'{folder}/checkpoints/training_log.csv').to_numpy()
        data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
        for i, column in enumerate(df0.columns):
            mean[column], std[column] = data_mean[:, i], data_std[:, i]
        pd.DataFrame(mean).to_csv(f'{self.train_kwargs["folder"]}/checkpoints/training_log.csv', index=None)
        pd.DataFrame(std).to_csv(f'{self.train_kwargs["folder"]}/checkpoints/training_log_std.csv', index=None)

        arr = np.zeros(self.n_exprs)
        for k, column in enumerate(df0.columns):
            for i in range(self.n_exprs):
                j = self.find_stable_stopping_point(data[i][:, int(9 + self.train_kwargs["forget_class"])], threhold)
                arr[i] = data[i, j, k]
            summary[column] = arr.mean() 
            summary_std[column] = arr.std() 

        # More stable estimate of FID
        if os.path.exists(self.train_kwargs['folder'] + '/fid.csv'):
            fid_score = pd.read_csv(self.train_kwargs['folder'] + '/fid.csv')["FID"].to_numpy()
            summary["FID"] = fid_score.mean()
            summary_std["FID"] = fid_score.std()

        summary["Time/Step"] = summary["Time"] + 0.
        summary["Time"] *= summary["Step"]

        summary_std["Time/Step"] = summary_std["Time"] + 0.
        summary_std["Time"] = np.sqrt(summary["Step"]**2 * summary_std["Time/Step"] **2 +\
                                      summary["Time/Step"] **2 * summary_std["Step"] **2 +\
                                      summary_std["Step"] **2 * summary_std["Time/Step"] **2)
        
        summary["Original FID"] = self.original_fid(num_fid_samples, device, batch_size)


        with open(f"{self.train_kwargs['folder']}/summary.json", 'w') as file:
            json.dump(summary, file, indent=2)

        with open(f"{self.train_kwargs['folder']}/summary_std.json", 'w') as file:
            json.dump(summary_std, file, indent=2)

        with open(f"{self.train_kwargs['folder']}/config.json", 'w') as file:
            json.dump(ut.get_config(self.get_folder(0)), file, indent=2)

        viz.evolve(self.train_kwargs['folder'])

    def summarize_wo_fid(self, threhold=2e-2):
        """
        Summarize the training curves of all experiments and save the statistics to summary.json and summary_std.json

        Parameters
        ----------
        threhold : float, optional
            The threshold to find the stable stopping point, by default 2e-2
        Returns
        -------
        None
        """
        if not os.path.exists(f'{self.train_kwargs["folder"]}/checkpoints'):
            os.makedirs(f'{self.train_kwargs["folder"]}/checkpoints')

        mean, std, summary, summary_std = {}, {}, {}, {}
        df0 = pd.read_csv(f'{self.get_folder(0)}/checkpoints/training_log.csv')
        arr = df0.to_numpy()
        data = np.zeros((self.n_exprs, arr.shape[0], arr.shape[1]))
        for i in range(self.n_exprs):
            folder = self.get_folder(i)
            data[i] = pd.read_csv(f'{folder}/checkpoints/training_log.csv').to_numpy()
        data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
        for i, column in enumerate(df0.columns):
            mean[column], std[column] = data_mean[:, i], data_std[:, i]
        pd.DataFrame(mean).to_csv(f'{self.train_kwargs["folder"]}/checkpoints/training_log.csv', index=None)
        pd.DataFrame(std).to_csv(f'{self.train_kwargs["folder"]}/checkpoints/training_log_std.csv', index=None)

        arr = np.zeros(self.n_exprs)
        for k, column in enumerate(df0.columns):
            for i in range(self.n_exprs):
                j = self.find_stable_stopping_point(data[i][:, int(9 + self.train_kwargs["forget_class"])], threhold)
                arr[i] = data[i, j, k]
            summary[column] = arr.mean() 
            summary_std[column] = arr.std() 

        # More stable estimate of FID
        if os.path.exists(self.train_kwargs['folder'] + '/fid.csv'):
            fid_score = pd.read_csv(self.train_kwargs['folder'] + '/fid.csv')["FID"].to_numpy()
            summary["FID"] = fid_score.mean()
            summary_std["FID"] = fid_score.std()

        summary["Time/Step"] = summary["Time"] + 0.
        summary["Time"] *= summary["Step"]

        summary_std["Time/Step"] = summary_std["Time"] + 0.
        summary_std["Time"] = np.sqrt(summary["Step"]**2 * summary_std["Time/Step"] **2 +\
                                      summary["Time/Step"] **2 * summary_std["Step"] **2 +\
                                      summary_std["Step"] **2 * summary_std["Time/Step"] **2)
        
        # summary["Original FID"] = self.original_fid(num_fid_samples, device, batch_size)


        # with open(f"{self.train_kwargs['folder']}/summary.json", 'w') as file:
        #     json.dump(summary, file, indent=2)

        # with open(f"{self.train_kwargs['folder']}/summary_std.json", 'w') as file:
        #     json.dump(summary_std, file, indent=2)

        # with open(f"{self.train_kwargs['folder']}/config.json", 'w') as file:
        #     json.dump(ut.get_config(self.get_folder(0)), file, indent=2)

        # viz.evolve(self.train_kwargs['folder']) 
        return summary, summary_std

    
    def original_fid(self, n_samples, device, batch_size=256):
        """
        Calculate the Fréchet Inception Distance (FID) between real and generated images using the original model.

        Parameters
        ----------
        n_samples : int
            The number of samples to be loaded from the dataset for FID computation.
        device : str or torch.device
            The device to perform computations on. If a string is provided, it will be converted to a torch.device.
        batch_size : int, optional
            The batch size for loading data. Defaults to 256.

        Returns
        -------
        float
            The computed FID score between the real and generated images.

        Notes
        -----
        This function loads the MNIST dataset and a pre-trained model to generate images.
        The FID score is calculated between the real images from the dataset and the generated images from the model.
        """

        if isinstance(device, str):
            device = torch.device(device)
        self.train_kwargs['data_path']
        dataloader = datapipe.CelebAData(img_path=self.train_kwargs['data_path']).get_dataloader(n_samples)
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
        
        # load models
        model = vt.init_model(model=self.train_kwargs['model'], latent_dim=self.train_kwargs['latent_dim'], device=device)

        with torch.no_grad():
            gen_images = model.decode(torch.randn(real_images.shape[0], self.train_kwargs['latent_dim']).to(device))
        
        return self.compute_fid(real_images, gen_images, batch_size)

    
    
    def find_stable_stopping_point(self, signal, threshold):
        """
        Finds the last index where the signal was above the threshold.

        Args:
            signal: Array-like time series.
            threshold: Value above which the signal is considered 'active'.

        Returns:
            Index of the last value in the signal that is > threshold.
            Returns -1 if no such point exists.
        """
        signal = np.asarray(signal)
        above_indices = np.where(signal > threshold)[0]
        return above_indices[-1] if len(above_indices) > 0 else -1
        

    
    @ut.timer
    def fid(self, n_samples, device, batch_size=256):
        """
        Calculate the Fréchet Inception Distance (FID) for all experiments in the folder.

        Parameters
        ----------
        n_samples : int
            Number of samples to use for FID computation.
        device : str or torch.device
            Device to use for computation. If str, it will be converted to a torch.device.
        batch_size : int, optional
            Batch size for loading data. Defaults to 256.

        Returns
        -------
        None

        Notes
        -----
        The function iterates over each experiment, computes the FID score using real images and generated images
        from the folder, and saves the results to a CSV file. If a CUDA device is used, it clears the cache after
        each experiment to free up memory. The function is decorated with @ut.timer, so it will print out the 
        time taken for execution.
        """

        if isinstance(device, str):
            device = torch.device(device)
        # attr_path = self.train_kwargs['data_path'] + '/list_attr_celeba.csv'
        dataloader = datapipe.CelebAData(img_path=self.train_kwargs['data_path']).get_dataloader(n_samples)
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
        # identifier = cl.get_classifier(self.train_kwargs['classifier_path'], device=device)
        
        # Prepare to collect FID scores
        results = [("expr-id", "FID")]
        for i in range(self.n_exprs):
            folder = self.get_folder(i)
            try:
                fid_score = self.compute_fid_from_folder(real_images, folder, batch_size)
            except:
                fid_score = np.nan
            results.append((i, fid_score))  # collect result
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Write to CSV
        with open(self.train_kwargs['folder'] + '/fid.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
    
    
    
    @ut.timer
    def compute_fid_from_folder(self, real_images, folder, batch_size=256):
        """
        Computes the Fréchet Inception Distance (FID) for the latest checkpoint in a given folder.

        Parameters
        ----------
        real_images : torch.Tensor
            A batch of real images for which to compute the FID.
        folder : str
            The path to the folder containing checkpoints of the model.
        batch_size : int, optional
            The number of images to process at once for feature extraction. Default is 256.

        Returns
        -------
        float
            The computed FID score between the real images and images generated by the model 
            from the latest checkpoint in the specified folder.

        Raises
        ------
        FileNotFoundError
            If no checkpoints are found in the specified folder.
        
        Notes
        -----
        The function loads the model from the latest checkpoint, generates images using a 
        random latent vector, and calculates the FID score using the real images and the 
        generated images. The function is decorated with @ut.timer to measure execution time.
        """

        device = real_images.device

        # Find the latest checkpoint
        checkpoints = glob.glob(os.path.join(folder, 'checkpoints', '*.pth'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {folder}/checkpoints/")
        
        def extract_int(path):
            return int(os.path.basename(path).split('_')[-1].split('.')[0])
        
        checkpoints.sort(key=extract_int)
        checkpoint = checkpoints[-1]

        # Load model
        model = torch.load(checkpoint, weights_only=False, map_location=device)
        model.to(device)
        model.eval()

        # Generate images
        with torch.no_grad():
            z = torch.randn(real_images.size(0), self.train_kwargs['latent_dim'], device=device)
            gen_images = model.decode(z)

        # Compute FID from tensors
        fid_score = self.compute_fid(real_images, gen_images, batch_size)

        # print(f"[{identifier}] FID: {fid_score:.2f}")
        return fid_score


 

    def compute_fid(self, real_images, gen_images, batch_size=256):
        """
        Computes the Frechet Inception Distance (FID) between two sets of images.

        Parameters
        ----------
        real_images : torch.Tensor
            The real images to use for FID computation.
        gen_images : torch.Tensor
            The generated images to use for FID computation.
        batch_size : int, optional
            The batch size to use for Inception feature extraction. Defaults to 256.

        Returns
        -------
        float
            The FID score between the real and generated images.
        """

        # 1) Extract features
        real_feats = compute_inception_features(real_images, batch_size)
        gen_feats  = compute_inception_features(gen_images, batch_size)

        # 2) Compute statistics
        mu_r = real_feats.mean(dim=0)
        mu_g = gen_feats.mean(dim=0)

        # Centered features
        real_c = real_feats - mu_r
        gen_c = gen_feats - mu_g

        # Compute covariances (unbiased, rowvar=False equivalent)
        cov_r = real_c.T @ real_c / (real_feats.shape[0] - 1)
        cov_g = gen_c.T @ gen_c / (gen_feats.shape[0] - 1)

        # 3) Symmetrize covariances
        cov_r = (cov_r + cov_r.T) / 2
        cov_g = (cov_g + cov_g.T) / 2

        # 4) Compute sqrt of product, with retries
        def sqrtm_psd(matrix):
            eigvals, eigvecs = eigh(matrix)
            eigvals = torch.clamp(eigvals, min=0)
            return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

        covmean = None
        for i in range(3):
            try:
                prod = cov_r @ cov_g
                prod = (prod + prod.T) / 2  # enforce symmetry
                covmean = sqrtm_psd(prod)
                if not torch.isfinite(covmean).all():
                    raise ValueError("Non-finite result")
                break
            except Exception:
                jitter = torch.eye(cov_r.shape[0], device=real_images.device) * (1e-9 * (10**i))
                cov_r_j = cov_r + jitter
                cov_g_j = cov_g + jitter
                prod = (cov_r_j @ cov_g_j + (cov_r_j @ cov_g_j).T) / 2
                covmean = sqrtm_psd(prod)

        # 5) Final FID formula
        covmean = (covmean + covmean.T) / 2  # ensure symmetry
        diff = mu_r - mu_g
        fid = diff @ diff + torch.trace(cov_r + cov_g - 2 * covmean)

        return float(fid)



class BatchCompare:
    def __init__(self, folders, labels):
        """
        Constructor for BatchCompare.
        
        Parameters
        ----------
        folders : list
            List of folders containing experiment results.
        labels : list
            List of labels for the folders.
        
        Attributes
        ----------
        folders : list
            List of folders containing experiment results.
        labels : list
            List of labels for the folders.
        summary : list
            List of summary dictionaries for the folders.
        summary_std : list
            List of summary standard deviation dictionaries for the folders.
        """
        self.folders = folders
        self.labels = labels 
        self.summary = [self.get_summary(folder) for folder in folders]
        self.summary_std = [self.get_summary_std(folder) for folder in folders]

    def get_summary(self, folder):
        """
        Reads and returns the summary data from a JSON file.

        Parameters
        ----------
        folder : str
            The path to the folder containing the summary.json file.

        Returns
        -------
        dict
            A dictionary containing the summary data.
        """
        with open(f"{folder}/summary.json", 'r') as file:
            return json.load(file) 
    
    
    def get_summary_std(self, folder):
        """
        Reads and returns the summary standard deviation data from a JSON file.

        Parameters
        ----------
        folder : str
            The path to the folder containing the summary_std.json file.

        Returns
        -------
        dict
            A dictionary containing the summary standard deviation data.
        """
        
        with open(f"{folder}/summary_std.json", 'r') as file:
            return json.load(file)

    def plot(self, *args):
        """
        Plots the mean and standard deviation of various metrics for each folder.

        Parameters
        ----------
        *args : str
            Additional metrics to plot in the second row of subplots.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        self.plot_mean_std(ax[0, 0], "Time", flag='min')
        self.plot_mean_std(ax[0, 1], "FID", log=True, flag='min')
        self.plot_mean_std(ax[0, 2], "Margin")
        for i, arg in enumerate(args):
            self.plot_mean_std(ax[1, i], arg)
        return fig, ax

    def plot2(self, logs=[False, True], rank=False):
        """
        Plots the mean and standard deviation of time and FID across multiple folders.

        Parameters
        ----------
        rank : bool, optional
            If True, annotates each data point with its ranking. Defaults to False.
        logs : list, optional
            List of boolean values indicating whether to use a log scale for each subplot. Defaults to [False, True].

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        self.plot_mean_std(ax[0], "Time", log=logs[0], flag='min', rank=rank)
        self.plot_mean_std(ax[2], "FID", log=logs[1], flag='min', rank=rank)
        # fig.subplots_adjust(wspace=0.3)
        try:
            ax[2].axhline(self.summary[0]['Original FID'], color='darkgrey', linestyle='--', label='Original model')
            ax[2].legend(fontsize=14, loc='lower right')
        except:
            pass
        for i, folder in enumerate(self.folders):
            df = self.average_training_log(folder)
            ax[1].plot(df['Step'] + 1, df['1 Fraction'], label=self.labels[i])
        # ax[1].axhline(0.04, color='black')
        ax[1].set_xlabel('Training step', fontsize=14)
        ax[1].set_ylabel('Fraction of glasses in generated images', fontsize=16)
        ax[1].legend(fontsize=14)
        ticks = [1, 100, 200, 300, 400, 500, 600]
        ax[1].set_xticks(ticks)  # choose suitable max_x and step
        ax[1].set_xticklabels([str(x) for x in ticks])
        ax[1].tick_params(axis='x', labelsize=14)

        for a in ax:
            a.tick_params(axis='y', labelsize=14)
        return fig, ax
        

    def plot_mean_std(self, ax, quantity, log=False, flag='min', rank=False):
        """
        Plots the mean and standard deviation of a given metric across multiple folders.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on.
        quantity : str
            The metric to plot.
        log : bool, optional
            If True, use a log scale for the y-axis. Defaults to False.
        flag : str, optional
            If 'min', marks the minimum value. If 'max', marks the maximum value. Defaults to 'min'.
        rank : bool, optional
            If True, annotates each data point with its ranking. Defaults to False.
        """
        y = [summary[quantity] for summary in self.summary]
        y_std = [std[quantity] for std in self.summary_std]
        y_std = [np.zeros_like(y_std), y_std]

        ax.errorbar(range(len(self.folders)), y, yerr=y_std)
        ax.scatter(range(len(self.folders)), y, s=50)
        ax.set_xticks(range(len(self.folders)))
        ax.set_xticklabels(self.labels, fontsize=14)
        ax.set_ylabel(f"{quantity}" if quantity != 'Time' else 'Time to unlearn (s)', fontsize=16)
        if log:
            ax.set_yscale('log')
        # if flag == 'min':
        #     ax.scatter(y.index(min(y)), min(y), marker='x', color='red', s=100)
        # elif flag == 'max':
        #     ax.scatter(y.index(max(y)), max(y), marker='x', color='red', s=100)

        if rank:
            rankings = self.compute_rankings(y, flag=flag)
            print([std[quantity] for std in self.summary_std])
            y = np.array(y)
            y_range = y.max() - y.min()
              # ensure minimum offset

            # for i, (x, y_val, rank) in enumerate(zip(range(len(y)), y, rankings)):
            #     y_offset = 0.3 * max(abs(y_val), 1e-3) if log else 0.02 * y_range
            #     ax.text(
            #         x, y_val - y_offset,
            #         f'{rank}',
            #         ha='center', va='bottom', fontsize=13,
            #         bbox=dict(facecolor='white', edgecolor='none', pad=0.0, alpha=1.0)
            #     )
            xtick_labels = [f"{label}\n({rank})" for label, rank in zip(self.labels, rankings)]
            ax.set_xticklabels(xtick_labels, fontsize=14)
        min_y = np.min(y)
        ax.set_ylim(bottom=min_y * 0.8)
    
    

    def compute_rankings(self, y, flag='min'):
        """
        Computes the rankings of the given array `y` with rounding to 3 decimals after 0.
        Identical values after rounding receive identical ranks.

        Parameters
        ----------
        y : array_like
            The array to compute rankings for.
        flag : str, optional
            If 'min', ranks in ascending order. If 'max', ranks in descending order. Defaults to 'min'.

        Returns
        -------
        ranks : ndarray
            The rankings of `y` with ties based on rounded values.
        """
        y = np.array(y)
        
        # Round to 3 decimals after 0
        rounded = np.round(y, 3)
        
        # Get unique values in sorted order
        if flag == 'min':
            unique_vals = np.sort(np.unique(rounded))
        else:
            unique_vals = np.sort(np.unique(rounded))[::-1]

        # Map each unique value to a rank
        val_to_rank = {val: rank + 1 for rank, val in enumerate(unique_vals)}

        # Assign ranks based on rounded values
        ranks = np.array([val_to_rank[val] for val in rounded])

        return ranks
        

    def average_training_log(self, folder, relative_path='checkpoints/training_log.csv'):
        """
        Reads training_log.csv from each folder and computes the average over all.

        Parameters
        ----------
        folder : str
            Path to the folder containing the training logs.
        relative_path : str
            Relative path to the training log inside each folder.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the average of the training logs.
        """
        dfs = []
        folders = self.get_expr_folders(folder)
        for folder in folders:
            path = os.path.join(folder, relative_path)
            df = pd.read_csv(path)
            dfs.append(df)

        # Ensure all DataFrames have the same length and columns
        for i, df in enumerate(dfs):
            if not df.columns.equals(dfs[0].columns) or len(df) != len(dfs[0]):
                raise ValueError(f"Inconsistent format or length in {folders[i]}")

        # Stack and average
        stacked = pd.concat(dfs, axis=0, keys=range(len(dfs)))  # multi-index [replicate, row]
        mean_df = stacked.groupby(level=1).mean()  # average over folders

        return mean_df
    

    def get_expr_folders(self, parent_dir):
        """
        Returns a list of subfolders in parent_dir that start with 'expr-'.

        Parameters
        ----------
        parent_dir : str
            The path to the directory to search.

        Returns
        -------
        list of str
            List of full paths to subfolders starting with 'expr-'.
        """
        return [
            os.path.join(parent_dir, name)
            for name in os.listdir(parent_dir)
            if name.startswith('expr-') and os.path.isdir(os.path.join(parent_dir, name))
        ]