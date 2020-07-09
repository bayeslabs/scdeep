# DeepImpute implementation in PyTorch

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import expon
import torch.optim as optim
from typing import Union

from scdeep.trainer import Trainer
from scdeep.dataset import GeneExpressionDataset


class SubModule(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(SubModule, self).__init__()
        self.dense_layer = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.dense_layer(x))
        x = self.dropout(x)
        x = F.softplus(self.output_layer(x))
        return x


class DeepImputeModel(nn.Module):

    def __init__(self):
        super(DeepImputeModel, self).__init__()
        self.inp = []
        self.module_list = nn.ModuleList([])

    def initialize_modules(self, input_dims):
        self.inp = input_dims
        for i in self.inp:
            self.module_list.append(SubModule(input_dim=i))

    def forward(self, x):
        # x = torch.split(x, self.inp, dim=1)
        output = []
        for i, mod in enumerate(self.module_list):
            output.append(mod(x[i]))
        output = torch.cat(output, dim=1)
        return output


class DeepImputeTrainer(Trainer):

    def __init__(
            self,
            model: DeepImputeModel,
            gene_dataset: GeneExpressionDataset,
            genes_to_impute=None,
            min_vmr=0.5,
            nn_lim="auto",
            number_predictor=None,
            n_top_correlated_genes=5,
            subset_dim=512,
            mode="random",
            **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)

        self.targets = []
        self.predictors = []
        self.min_vmr = min_vmr
        self.subset_dim = subset_dim

        if type(self.gene_dataset.data) == np.ndarray or type(self.gene_dataset.data) == scipy.sparse.csr_matrix:
            self.gene_dataset.data = pd.DataFrame(self.gene_dataset.data, dtype='float32')
        else:
            self.gene_dataset.data = pd.DataFrame(self.gene_dataset.data.values, dtype='float32')
        # (note: below operations are carried out on data assuming it is a pandas dataframe)
        genes_vmr = (self.gene_dataset.data.var() / (1 + self.gene_dataset.data.mean())).sort_values(ascending=False)
        genes_vmr = genes_vmr[genes_vmr > 0]

        # In case 1, while filling genes, we repeat genes that have been previously selected
        # but do not choose genes that have a VMR < 0.5
        # In case 2, irrespective of VMR, we select genes from the already selected gene pool for filling genes
        if genes_to_impute is None:
            genes_to_impute = self.filter_genes(genes_vmr, min_vmr, nn_lim)
        else:
            number_genes = len(genes_to_impute)
            if number_genes % self.subset_dim != 0:
                number_fill = self.subset_dim - (number_genes % self.subset_dim)
                fill_genes = genes_to_impute[:number_fill]
                genes_to_impute = np.concatenate((genes_to_impute, fill_genes))

        self.corr_matrix = self.correlation_matrix(self.gene_dataset.data, number_predictor)

        # the  next two functions save the INDICES of genes that are to form the predictors and targets
        self.set_targets(self.gene_dataset.data.reindex(columns=genes_to_impute), mode)
        self.set_predictors(self.corr_matrix, n_top_correlated_genes)

        self.gene_dataset.data = np.log1p(self.gene_dataset.data.values).astype(np.float32)
        # self.gene_dataset.data = torch.tensor(self.gene_dataset.data.values)

        (train_set, test_set, val_set) = self.train_test_validation()
        self.register_posterior(train_set, test_set, val_set)

    def loss(self, y_pred, y_true):
        l = []
        for i, ytrue in enumerate(y_true):
            a = ytrue - y_pred[i]
            y = (ytrue * torch.mul(a, a))
            l.append(y.mean())
        return l

    def on_training_begin(self):
        self.model.initialize_modules([len(pred) for pred in self.predictors])
        self.model = self.model.cuda() if self.use_cuda else self.model
        for name, param in self.model.named_parameters():
            print(name)
            print(param.shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def model_output(self, data_tensor):
        data_tensors, indices = data_tensor
        inp = [data_tensors[:, column] for column in self.predictors]
        # inp = torch.split(data_tensors, [len(i) for i in self.predictors], dim=1)
        output = self.model(inp)
        output = torch.split(output, [len(t) for t in self.targets], dim=1)
        target = [data_tensors[:, column] for column in self.targets]
        return output, target

    def on_training_loop(self, data_tensor):
        output, target = self.model_output(data_tensor)
        self.current_loss = loss = self.loss(output, target)
        self.current_loss = torch.mean(torch.stack(loss))
        self.optimizer.zero_grad()
        for l in loss:
            l.backward(retain_graph=True)
        self.optimizer.step()

    @torch.no_grad()
    def on_validation(self, data_tensor, loss):
        output, target = self.model_output(data_tensor)
        loss.append(np.asarray([l.item() for l in self.loss(output, target)]).mean())
        return loss

    def correlation_matrix(self, data, number_predictor=None):

        # we find CV to find one value per gene to calculate the correlation between two genes
        # data = data.loc[:, ~data.columns.duplicated()]
        cv = data.std() / data.mean()
        cv[np.isinf(cv)] = 0
        if number_predictor is None:
            predictions = data.loc[:, cv > 0]
        else:
            predictions = data.loc[:, cv.sort_values(ascending=False).index[:number_predictor]]
        # the strongest linear relationship between two values
        # is represented by the absolute value of the Pearson's correlation coefficient
        corr_mat = pd.DataFrame(np.abs(np.corrcoef(predictions, rowvar=False)), index=predictions.columns,
                                columns=predictions.columns).fillna(0)
        return corr_mat

    def filter_genes(self, genes, min_vmr: Union[int, float], nn_lim):
        if not str(nn_lim).isdigit():
            nn_lim = (genes > min_vmr).sum()

        number_subsets = int(np.ceil(nn_lim / self.subset_dim))
        genes_to_impute = genes.index[:number_subsets * self.subset_dim]

        rest = self.subset_dim - (len(genes_to_impute) % self.subset_dim)
        if rest > 0 and rest != self.subset_dim:
            fill_genes = np.random.choice(genes.index, rest)
            genes_to_impute = np.concatenate((genes_to_impute, fill_genes))
        # genes_to_impute contains the indices of genes that should be included for imputation
        return genes_to_impute

    def set_targets(self, data, mode):  # mode = random / progressive

        number_subsets = int(data.shape[1] / self.subset_dim)
        if mode == 'progressive':
            self.targets = data.columns.values.reshape((number_subsets, self.subset_dim))
        else:
            self.targets = np.random.choice(data.columns, size=(number_subsets, self.subset_dim), replace=False)
        # self.targets = list(self.targets)

    def set_predictors(self, covariance_matrix, n_top_correlated_genes):
        self.predictors = []
        for i, target in enumerate(self.targets):
            predictor = covariance_matrix.loc[target].drop(np.intersect1d(target, covariance_matrix.columns), axis=1)
            sorted_args = np.argsort(-predictor.values, axis=1)[:, :n_top_correlated_genes]
            predictor = predictor.columns[sorted_args.flatten()]

            self.predictors.append(np.unique(predictor))

            print("Network {}: {} predictors, {} targets".format(i, len(np.unique(predictor)), len(target)))

    def get_probs(self, vec, distr):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(distr)

    def mask_data(self, data_to_mask, test_size, distr="exp", dropout=0.01):
        np.random.seed(self.seed)

        permuted_indices = np.random.permutation(data_to_mask.shape[0])
        test_set = int(np.ceil(data_to_mask.shape[0] * test_size))
        test_indices = np.array(permuted_indices[:test_set])

        data_to_mask = data_to_mask[test_indices, :]
        bin_mask = np.ones(data_to_mask.shape).astype(bool)

        for c in range(data_to_mask.shape[0]):
            cells = data_to_mask[c, :]
            positive_indices = np.arange(data_to_mask.shape[1])[cells > 0]
            positive_cells = cells[positive_indices]

            if positive_cells.size > 5:
                probs = self.get_probs(positive_cells, distr)
                n_masked = 1 + int(dropout * len(positive_cells))
                if n_masked >= positive_cells.size:
                    print("Warning: too many cells masked")
                    n_masked = 1 + int(0.5 * len(positive_cells))
                masked_idx = np.random.choice(
                    positive_cells.size, n_masked, p=probs / probs.sum(), replace=False
                )
                bin_mask[c, positive_indices[sorted(masked_idx)]] = False
        unmasked_dataset = np.copy(data_to_mask)
        data_to_mask[~bin_mask] = 0
        return unmasked_dataset, data_to_mask, test_indices

    @torch.no_grad()
    def predict(self, data, return_imputed_only=False, policy="restore"):

        data_tensor = torch.tensor(data.values, device=self.device)

        inp = [data_tensor[:, column] for column in self.predictors]

        predicted = self.model(inp)

        predicted = pd.DataFrame(predicted.cpu().detach().numpy(), index=data.index, columns=self.targets.flatten())
        predicted = predicted.groupby(by=predicted.columns, axis=1).mean()
        not_predicted = data.drop(columns=self.targets.flatten())
        imputed = pd.concat([predicted, not_predicted], axis=1).loc[data.index, data.columns]

        if policy == "restore":
            not_replaced_values = (data > 0)
            imputed[not_replaced_values] = data[not_replaced_values]
        elif policy == "max":
            not_replaced_values = (data > imputed)
            imputed[not_replaced_values] = data[not_replaced_values]

        return imputed

    def model_score(self, data, test_size=0.2):

        data, masked_dataset, row_indices = self.mask_data(data, test_size=test_size)
        data = pd.DataFrame(data, index=row_indices)
        masked_dataset = pd.DataFrame(masked_dataset, index=row_indices)

        imputed = self.predict(masked_dataset)

        imputed_genes = np.intersect1d(data.columns, imputed.columns)
        masked_index = data[imputed_genes].values != masked_dataset[imputed_genes].values

        score = self.metric_score(data[imputed_genes].values[masked_index], imputed[imputed_genes].values[masked_index])
        return score

    def metric_score(self, a, b):
        return scipy.stats.pearsonr(a.reshape(-1), b.reshape(-1))[0]







