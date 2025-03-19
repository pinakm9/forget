import pandas as pd
import numpy as np
import json
import utility as ut
import vae_viz as viz
import os


class BatchExperiment:
    def __init__(self, train_func, train_kwargs, n_exprs):
        self.train_func = train_func
        self.train_kwargs = train_kwargs
        self.run(n_exprs)    

    @ut.timer
    def run(self, n_exprs):
        self.n_exprs = n_exprs
        for i in range(n_exprs):
            train_kwargs = self.train_kwargs.copy()
            train_kwargs['folder'] = self.get_folder(i)
            self.train_func(**train_kwargs)
        self.summarize()

    def get_folder(self, index):
        return f'{self.train_kwargs["folder"]}/expr-{index}'

    def summarize(self):
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
                j = np.argmax(data[i][:, int(11 + self.train_kwargs["forget_digit"])] < 2e-2)
                arr[i] = data[i, j, k]
            summary[column] = arr.mean() 
            summary_std[column] = arr.std() 

        summary["Time/Step"] = summary["Time"] + 0.
        summary["Time"] *= summary["Step"]

        summary_std["Time/Step"] = summary_std["Time"] + 0.
        summary_std["Time"] = np.sqrt(summary["Step"]**2 * summary_std["Time/Step"] **2 +\
                                      summary["Time/Step"] **2 * summary_std["Step"] **2 +\
                                      summary_std["Step"]**2 * summary_std["Time/Step"] **2)

        with open(f"{self.train_kwargs['folder']}/summary.json", 'w') as file:
            json.dump(summary, file, indent=2)

        with open(f"{self.train_kwargs['folder']}/summary_std.json", 'w') as file:
            json.dump(summary_std, file, indent=2)

        with open(f"{self.train_kwargs['folder']}/config.json", 'w') as file:
            json.dump(ut.get_config(self.get_folder(0)), file, indent=2)

        viz.evolve(self.train_kwargs['folder']) 
            