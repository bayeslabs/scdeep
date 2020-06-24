# The structure of the dataset, posterior and trainer classes are based on the structures used in ScVI
# The implementation of ScVI by YosefLab is available on GitHub

import numpy as np
import pandas as pd
import scipy.sparse
from torch.utils.data import Dataset
import scanpy

from typing import Union, Dict, List
from collections import OrderedDict, defaultdict
import logging

logger = logging.getLogger(__name__)


class GeneExpressionDataset(Dataset):

    def __init__(self):

        # registers
        self.gene_attribute_names = set()
        self.cell_attribute_names = set()
        self.attribute_mappings = defaultdict(list)

        # attributes
        self._data = None
        self._raw = None
        self._batch_indices = None
        self._labels = None
        self.num_batches = None
        self.num_labels = None
        self.gene_names = None
        self.cell_types = None

    def __repr__(self): # todo finish this function
        pass

    def from_data(
            self,
            data: Union[np.ndarray, scipy.sparse.csr_matrix],
            raw: Union[np.ndarray, scipy.sparse.csr_matrix] = None,
            batch_indices: Union[List[int], np.ndarray, scipy.sparse.csr_matrix] = None,
            labels: Union[List[int], np.ndarray, scipy.sparse.csr_matrix] = None,
            gene_names: Union[List[str], np.ndarray] = None,
            cell_types: Union[List[str], np.ndarray] = None,
            cell_attributes_dict: Dict[str, Union[np.ndarray, List]] = None,
            gene_attributes_dict: Dict[str, Union[np.ndarray, List]] = None
                ):
        # todo write description of parameters

        # data = data.T
        self._data = (
            np.ascontiguousarray(data, dtype=np.float32)
            if isinstance(data, np.ndarray)
            else data
        )

        self._raw = (
            np.ascontiguousarray(raw, dtype=np.float32)
            if isinstance(raw, np.ndarray)
            else raw
        )

        self.initialize_cell_attribute(
            "batch_indices",
            np.asarray(batch_indices).reshape((-1, 1))
            if batch_indices is not None
            else np.zeros((data.shape[0], 1))
        )

        self.initialize_cell_attribute(
            "labels",
            np.ndarray(labels).reshape((-1, 1))
            if labels is not None
            else np.zeros((data.shape[0], 1))
        )

        if gene_names is not None:
            genes = np.asarray(gene_names, dtype="<U64")
            self.initialize_gene_attribute("gene_name", genes)
            if len(np.unique(self.gene_names)) != len(self.gene_names):
                logger.warning("Gene names are not unique.")

        if cell_types is not None:
            self.initialize_mapped_attribute(
                "labels", "cell_types", np.asarray(cell_types, dtype = "<U128")
            )
        elif labels is None:
            self.initialize_mapped_attribute(
                "labels", "cell_types", np.asarray(["undefined"], dtype = "<U128")
            )

        if cell_attributes_dict:
            for attribute_name, attribute_value in cell_attributes_dict.items():
                self.initialize_cell_attribute(attribute_name, attribute_value)
        if gene_attributes_dict:
            for attribute_name, attribute_value in gene_attributes_dict.items():
                self.initialize_gene_attribute(attribute_name, attribute_value)

    def initialize_cell_attribute(self, attribute_name, attribute):

        try:
            len_attribute = attribute.shape[0]
        except AttributeError:
            len__attribute = len(attribute)

        if not self.nb_cells == len_attribute:
            raise ValueError(
                "Number of cells ({number_cells}) and length of attribute ({number_attributes})"
                "are not the same.".format(number_cells=self.nb_cells, number_attributes=len_attribute)
            )
        setattr(
            self,
            attribute_name,
            np.asarray(attribute)
            if not isinstance(attribute, scipy.sparse.csr_matrix)
            else attribute
        )
        self.cell_attribute_names.add(attribute_name)

    def initialize_gene_attribute(self, attribute_name, attribute_value):
        if not self.nb_genes == len(attribute_value):
            raise ValueError(
                "Number of genes ({n_genes}) and length of gene attribute ({n_attr}) mismatch".format(
                    n_genes=self.nb_genes, n_attr=len(attribute_value)
                )
            )
        setattr(self, attribute_name, attribute_value)
        self.gene_attribute_names.add(attribute_name)

    def initialize_mapped_attribute(self, source_attribute_name, mapped_name, mapped_value):
        source_attribute = getattr(self, source_attribute_name)
        if isinstance(source_attribute, np.ndarray):
            source_type = source_attribute.dtype
        else:
            source_type = type(source_attribute[0])
        if not np.issubdtype(source_type, np.integer):
            raise ValueError(
                "The mapping (source) attribute ({source_name}) should be categorical "
                "and hence integer and not ({given_type})".format(source_name=source_attribute_name, given_type=source_type)
            )
        max_categorical_value = np.max(source_attribute)
        if not max_categorical_value <= len(mapped_value):
            raise ValueError(
                "The maximum categorical value ({max_val}) is less than mapped values ({len_mapped})".format(
                    max_val=max_categorical_value, len_mapped=len(mapped_value)
                )
            )
        self.attribute_mappings[source_attribute_name].append(mapped_name)
        setattr(self, mapped_name, mapped_value)

    # REGISTERS AND PROPERTIES

    def __len__(self):
        return self.nb_cells

    def __getitem__(self, item):
        return item

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]):
        num_dim = len(data.shape)
        if num_dim != 2:
            raise ValueError(
                "Gene Expression data should have 2 dimensions and not {}".format(num_dim)
            )
        if type(data) == scipy.sparse.coo_matrix:
            data = scipy.sparse.csr_matrix(data)
        self._data = data

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, raw: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]):
        num_dim = len(raw.shape)
        if num_dim != self._data.size:
            raise ValueError(
                "Gene Expression raw data should have matching dimensions as data."
            )
        if type(raw) == scipy.sparse.coo_matrix:
            raw = scipy.sparse.csr_matrix(raw)
        self._raw = raw

    @property
    def nb_cells(self) -> int:
        return self.data.shape[0]

    @property
    def nb_genes(self) -> int:
        return self.data.shape[1]

    @property
    def nb_cell_counts(self) -> np.ndarray:
        return np.sum(self.data, axis=1)

    @property
    def batch_indices(self) -> np.ndarray:
        return self._batch_indices

    @batch_indices.setter
    def batch_indices(self, batch_indices: Union[List[int], np.ndarray]):
        batch_indices = np.asarray(batch_indices, dtype=np.int64).reshape((-1, 1))
        self.num_batches = len(np.unique(batch_indices))
        self._batch_indices = batch_indices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, labels: Union[List[int], np.ndarray]):
        labels = np.asarray(labels, dtype=np.int64)
        self.num_labels = len(np.unique(labels))
        self._labels = labels


def normalize(
        adata: scanpy.AnnData,
        filter_min_counts=None,
        size_factors=False,
        scale_input=False,
        logtrans_input=False
):

    if filter_min_counts:
        scanpy.pp.filter_genes(adata, min_counts=filter_min_counts)
        scanpy.pp.filter_cells(adata, min_counts=filter_min_counts)

    adata.raw = adata
    dataset = GeneExpressionDataset()
    dataset.from_data(adata.X, raw=adata.raw.X)

    if size_factors:
        scanpy.pp.normalize_per_cell(adata)
        size_factor_cell = dataset.nb_cell_counts / np.median(dataset.nb_cell_counts)
        dataset.initialize_cell_attribute("size_factor", size_factor_cell.reshape((-1,1)))
    if logtrans_input:
        scanpy.pp.log1p(adata)
    if scale_input:
        scanpy.pp.scale(adata)

    dataset.data = adata.X
    obs_index = adata.obs.columns
    for i, index in enumerate(obs_index):
        dataset.initialize_cell_attribute(index, adata.obs.iloc[:,i].values)
    return dataset


