# The structure of the dataset, posterior and trainer classes are based on the structures used in ScVI
# The implementation of ScVI by YosefLab is available on GitHub

import logging
import numpy as np
import torch
import copy

from torch.utils.data import DataLoader, SubsetRandomSampler
from scdeep.dataset import GeneExpressionDataset

logger = logging.getLogger(__name__)


class Posterior:

    def __init__(
            self,
            model,
            gene_dataset: GeneExpressionDataset,
            shuffle=True,
            indices=None,
            use_cuda=True,
            data_loader_kwargs=dict()
    ):
        self.model = model
        self.gene_dataset = gene_dataset
        self.use_cuda = use_cuda

        if indices is not None:
            self.sampler = SubsetRandomSampler(indices)
        else:
            self.sampler = None

        self.data_loader_kwargs = copy.deepcopy(data_loader_kwargs)
        self.data_loader_kwargs.update({"sampler": self.sampler})
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    @property
    def num_cells(self):
        # if hasattr(self.data_loader.sampler, "indices"):
        #     return len(self.data_loader.sampler.indices)
        # else:
        #   return len(self.gene_dataset)
        return len(self.gene_dataset)

    @property
    def indices(self) -> np.ndarray:
        # if hasattr(self.data_loader.sampler, "indices"):
        #     return self.data_loader.sampler.indices
        # else:
        #   return np.arange(len(self.gene_dataset))
        return np.arange(len(self.gene_dataset))

    def __len__(self):
        return self.num_cells

    def __iter__(self):
        return map(self.to_cuda, [(torch.tensor(self.gene_dataset.data[i,:]),i) for i in iter(self.data_loader)])

    def to_cuda(self, t):
        return t[0].cuda() if self.use_cuda else t[0], t[1]







