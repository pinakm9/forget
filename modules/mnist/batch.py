import pandas as pd
import numpy as np
import json
import utility as ut
import vae_viz as viz
import os
import matplotlib.pyplot as plt 
import torch
from torch.linalg import eigh, svd
import classifier as cl
import datapipe, csv
import torch.nn.functional as F
import glob
import vae_train as vt

class BatchExperiment:
    def __init__(self, train_func, train_kwargs, n_exprs):
        """
        Constructor for BatchExperiment. 

        Parameters
        ----------
        train_func: function
            function to call to train VAE
        train_kwargs: dict
            arguments to be passed to train_func
        n_exprs: int
            number of times to run train_func
        """
        self.train_func = train_func
        self.train_kwargs = train_kwargs
        # self.run(n_exprs)    
        self.n_exprs = n_exprs

    @ut.timer
    def run(self):
        """
        Run the training loop in parallel over n_exprs experiments

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
        Return the folder name for the i-th experiment.

        Parameters
        ----------
        index: int
            Index of the experiment

        Returns
        -------
        str
            Folder name of the experiment
        """
        
        return f'{self.train_kwargs["folder"]}/expr-{index}'

    def summarize(self, threshold=2e-2, num_fid_samples=10000, device='cuda'):
        """
        Summarize the training curves of all experiments and save the statistics to summary.json and summary_std.json

        Parameters
        ----------
        threshold : float, optional
            The threshold to find the stable stopping point, by default 2e-2
        fid_kwargs : dict, optional

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
                j = self.find_stable_stopping_point(data[i][:, int(9 + self.train_kwargs["forget_digit"])], threshold)
                arr[i] = data[i, j, k] if column != "Time" else data[i, :, k].mean()
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
        
        summary["Original FID"] = self.original_fid(num_fid_samples, device)

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
                j = self.find_stable_stopping_point(data[i][:, int(9 + self.train_kwargs["forget_digit"])], threhold)
                arr[i] = data[i, j, k] if column != "Time" else data[i, :, k].mean()
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
        
        for filename, payload in (("summary.json", summary), ("summary_std.json", summary_std)):
            path = f"{self.train_kwargs['folder']}/{filename}"
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    prev = json.load(f)
                if not isinstance(prev, dict):
                    prev = {}
            except Exception:
                prev = {}
            for k, v in payload.items():
                if "FID" in k:
                    continue
                # ensure JSON-serializable (convert numpy scalars)
                if isinstance(v, (np.floating, np.integer)):
                    v = float(v)
                prev[k] = v
            with open(path, "w") as f:
                json.dump(prev, f, indent=2) 
        return summary, summary_std



    def original_fid(self, n_samples, device):
        """
        Calculate the Fréchet Inception Distance (FID) between real and generated images using the original model.

        Parameters
        ----------
        n_samples : int
            The number of samples to be loaded from the dataset for FID computation.
        device : str or torch.device
            The device to perform computations on. If a string is provided, it will be converted to a torch.device.

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
        dataloader = datapipe.MNIST().get_dataloader(n_samples, root=self.train_kwargs['data_path'], all_digits=self.train_kwargs['all_digits'])
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
        
        # load models
        model = vt.init_model(model=self.train_kwargs['model'], latent_dim=self.train_kwargs['latent_dim'], device=device)
        identifier = cl.get_classifier(self.train_kwargs['classifier_path'], device=device)

        with torch.no_grad():
            gen_images = model.decoder(torch.randn(real_images.shape[0], self.train_kwargs['latent_dim']).to(device))
        
        return self.compute_fid(real_images, gen_images, identifier)
        


    def find_stable_stopping_point(self, signal, threshold):
        """
        Finds the first index where the signal is below the threshold.
        """
        signal = np.asarray(signal)
        below_indices = np.where(signal < threshold)[0]
        return below_indices[0] if len(below_indices) > 0 else -1
    

    @ut.timer
    def fid(self, n_samples, device):
        """
        Compute the FID for all experiments in the folder.

        Parameters
        ----------
        n_samples : int
            Number of samples to use for FID computation.
        device : str or torch.device
            Device to use for computation. If str, it will be converted to a torch.device.

        Returns
        -------
        None

        Notes
        -----
        This function is decorated with @ut.timer, so it will print out the time taken.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.train_kwargs['data_path']
        dataloader = datapipe.MNIST().get_dataloader(n_samples, root=self.train_kwargs['data_path'], all_digits=self.train_kwargs['all_digits'])
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
        identifier = cl.get_classifier(self.train_kwargs['classifier_path'], device=device)
        
        # Prepare to collect FID scores
        results = [("expr-id", "FID")]
        for i in range(self.n_exprs):
            folder = self.get_folder(i)
            try:
                fid_score = self.compute_fid_from_folder(real_images, identifier, folder)
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
    def compute_fid_from_folder(self, real_images, identifier, folder):
        """
        Computes the FID of a given folder, containing a VAE model and a classifier.

        Parameters
        ----------
        real_images : torch.Tensor
            The real images to use for FID computation.
        identifier : cl.Classifier
            The classifier to use for FID computation.
        folder : str
            The folder containing the VAE model and classifier.

        Returns
        -------
        float
            The FID score of the given folder.
        """
        device = real_images.device 
        # find the last checkpoint
        checkpoints = glob.glob(''.join([folder, '/checkpoints/*.pth']))
        def extract_int(path):
            return int(os.path.basename(path).split('_')[-1].split('.')[0])           
            
        checkpoints.sort(key=extract_int)
        checkpoint = checkpoints[-1]

        # load models
        model = torch.load(checkpoint, weights_only=False, map_location=device)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            gen_images = model.decoder(torch.randn(real_images.shape[0], self.train_kwargs['latent_dim']).to(device))
        # compute fid
        return self.compute_fid(real_images, gen_images, identifier)

    
    def compute_features(self, images, identifier, layer=None):
        """
        Extract features from a single image or batch using a forward hook.

        Args:
            identifier (nn.Module): identifier instance (e.g., CNN)
            images (Tensor): input image tensor, shape [C, H, W] or [B, C, H, W]
            layer (nn.Module, optional): target layer to hook. Defaults to identifier.classifier[1].

        Returns:
            np.ndarray: feature vector(s), shape [1, D] or [B, D]
        """
        images = images.view(images.size(0), -1)
        features = torch.relu(F.linear(images, identifier.fc1.weight, identifier.fc1.bias))
        return features.detach()
        
 

    def compute_fid(self, real_images, gen_images, identifier):
        """
        Computes the Frechet Inception Distance (FID) between real and generated images.

        This function extracts features from both real and generated images, computes their 
        means and covariances, and then calculates the FID score using the squared difference 
        between the mean feature vectors and the trace of the sum of the covariances minus 
        twice the matrix square root of their product.

        Parameters
        ----------
        real_images : torch.Tensor
            The batch of real images.
        gen_images : torch.Tensor
            The batch of generated images.
        identifier : nn.Module
            A model used to extract features from the images.

        Returns
        -------
        float
            The computed FID score between the real and generated images.
        """
        # 1) Extract features
        
        real_feats = self.compute_features(real_images, identifier) # (N, D)
        gen_feats  = self.compute_features(gen_images, identifier)  # (M, D)

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
            ax[2].legend(fontsize=14, loc='upper right')
        except:
            pass
        for i, folder in enumerate(self.folders):
            df = self.average_training_log(folder)
            ax[1].plot(df['Step'] + 1, df['1 Fraction'], label=self.labels[i])
        ax[1].set_xlabel('Training step', fontsize=14)
        ax[1].set_ylabel('Fraction of 1\'s in generated images', fontsize=16)
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
    
    

    def compute_rankings(self,y, flag='min'):
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