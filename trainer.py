# The structure of the dataset, posterior and trainer classes are based on the structures used in ScVI
# The implementation of ScVI by YosefLab is available on GitHub

import logging
import time
from typing import Union, List
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from posterior import Posterior
from dataset import GeneExpressionDataset

logger = logging.getLogger(__name__)

# todo change the train size and test size functions and add a predict function instead and a test function

class Trainer:
    default_metrics_to_monitor = []

    def __init__(
            self,
            model,
            gene_dataset: GeneExpressionDataset,
            use_cuda: bool = True,
            metrics_to_monitor: List = None,
            frequency_stats: int = None,
            weight_decay: float = None,
            data_loader_kwargs: dict = None,
            show_progressbar: bool = True,
            batch_size: int = 128,
            train_size: Union[int, float] = 0.8,
            test_size: Union[int, float] = None,
            shuffle: bool = True,
            seed: int = 0,
            max_nans: int = 15,
            output_dir=None
    ):

        # model and dataset
        self.model = model
        self.gene_dataset = gene_dataset
        self.seed = seed
        self.use_cuda = torch.cuda.is_available() and use_cuda
        if self.use_cuda:
            model.cuda()

        # dataloader
        self._posterior = OrderedDict()
        self.batch_size = batch_size
        self.data_loader_kwargs = {"batch_size": batch_size, "pin_memory": use_cuda}
        data_loader_kwargs = data_loader_kwargs if data_loader_kwargs else dict()
        self.data_loader_kwargs.update(data_loader_kwargs)

        # optimization
        self.optimizer = None
        self.weight_decay = weight_decay
        self.num_epochs = None
        self.lr = None
        self.epoch = -1
        self.num_iter = 0
        self.training_time = 0

        # training nans
        self.max_nans = max_nans
        self.current_loss = None
        self.was_previous_loss_nan = False
        self.nan_count = 0

        self.output_dir = output_dir
        self.train_size = train_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.frequency_stats = 1 if frequency_stats is None else frequency_stats
        # todo metrics and early stopping

        self.show_progressbar = show_progressbar
        self.compute_metrics_time = 0

    def train(self, num_epochs=500, lr=1e-3, params=None, **extra_kwargs):

        begin = time.time()
        self.model.train()

        # initialization of other model's parameters
        self.training_extras_init(**extra_kwargs)

        self.compute_metrics_time = 0
        self.num_epochs = num_epochs
        self.lr = lr

        # todo compute_metrics function if needed

        self.on_training_begin()

        for self.epoch in tqdm(
            range(num_epochs),
            desc="training",
            disable=not self.show_progressbar
        ):
            self.on_epoch_begin()

            for data_tensor in self.data_load_loop("train_set"):
                self.on_iteration_begin()
                self.on_training_loop(data_tensor)
                self.on_iteration_end()
            self.on_epoch_end()

        # todo compute_metrics and early stopping and saving best metrics

        self.model.eval()
        self.training_extras_end()

        self.training_time += (time.time() - begin) - self.compute_metrics_time
        self.on_training_end()

    def model_output(self, data_tensor):
        data, indices = data_tensor
        output = self.model(data)
        return output

    def on_training_loop(self, data_tensor):
        output = self.model_output(data_tensor)
        self.current_loss = loss = self.loss(output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def training_extras_init(self, **extra_kwargs):
        pass

    def training_extras_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_epoch_begin(self):
        self.num_iter = 0
        print("Epoch: {}".format(self.epoch + 1))

    def on_iteration_begin(self):
        pass

    def on_iteration_end(self):
        self.check_training_status()
        print("Iteration: {} Loss: {:.4f}".format(self.num_iter, self.current_loss.item()))
        self.num_iter += 1

    @torch.no_grad()
    def on_epoch_end(self):
        if (self.epoch % self.frequency_stats == 0) or self.epoch == 0 or self.epoch == self.num_epochs:
            self.model.eval()
            loss = []
            for data_tensor in self.data_load_loop(self.validation):
                output = self.model_output(data_tensor)
                loss.append(self.loss(output))
            print("Validation Loss: {:.4f}".format(np.asarray(loss).mean()))
            self.model.train()

    def on_training_end(self):
        pass

    def check_training_status(self):
        is_loss_nan = torch.isnan(self.current_loss).item()
        if is_loss_nan:
            self.nan_count += 1
            self.was_previous_loss_nan = True
        else:
            self.nan_count = 0
            self.was_previous_loss_nan = False

        if self.nan_count >= self.max_nans:
            raise ValueError(
                "Loss was NaN {} times".format(self.nan_count)
            )

    def data_load_loop(self, return_set):
        return self._posterior[return_set]

    def train_test_validation(
            self,
            model=None,
            gene_dataset=None,
            type_class=Posterior
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset

        train_size = self.train_size
        test_size = self.test_size
        shuffle = self.shuffle

        n = len(self.gene_dataset)

        if shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)
        train_set = int(np.ceil(n * train_size))
        if test_size is None:
            test_size = 1.0 - train_size
        test_set = int(np.ceil(n * test_size))

        train_indices = indices[:train_set]
        test_indices = indices[train_set:(train_set + test_set)]
        val_indices = indices[(train_set + test_set):]

        if len(val_indices) > 0:
            self.validation = "val_set"
            self.test_on_epoch_end = True
        else:
            self.validation = "test_set"
            self.test_on_epoch_end = False

        return (
            self.create_posterior(
                model, gene_dataset, shuffle=shuffle, indices=train_indices, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, shuffle=shuffle, indices=test_indices, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, shuffle=shuffle, indices=val_indices, type_class=type_class
            )
        )

    def create_posterior(
            self,
            model=None,
            gene_dataset=None,
            shuffle=True,
            indices=None,
            type_class=Posterior
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset

        return type_class(
            model,
            gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs
        )

    def register_posterior(self, train_set, test_set, val_set):
        self._posterior["train_set"] = train_set
        self._posterior["test_set"] = test_set
        self._posterior["val_set"] = val_set

    def save_checkpoint(self):
        if not os.path.isdir(self.output_dir):
            os.system("mkdir -p {}".format(self.output_dir))
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.current_loss
        }, self.output_dir)

    def load_checkpoint(self):
        if not os.path.isdir(self.output_dir):
            raise ValueError("File path incorrect. Please specify correct path")
        checkpoint_dict = torch.load(self.output_dir)
        self.epoch = checkpoint_dict["epoch"]
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        self.current_loss = checkpoint_dict["loss"]
        self.model.eval()
