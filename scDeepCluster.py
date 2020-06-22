import numpy as np
import pandas as pd
from typing import List, Union
import logging
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans


from trainer import Trainer
from dataset import GeneExpressionDataset, normalize
from network import AutoEncoder, NBAutoEncoder, ZINBAutoEncoder
from losses import nb_loss, zinb_loss


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class ClusteringLayer(nn.Module):

    def __init__(
            self,
            input_dim,
            n_clusters: int,
            alpha=1.0,
            initial_weights=None,
    ):
        super(ClusteringLayer, self).__init__()
        self.n_features = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = initial_weights

        if self.initial_weights is None:
            weights = torch.tensor((self.n_clusters, self.n_features))
            nn.init.xavier_uniform_(weights)
            self.clusters = nn.Parameter(weights)
        else:
            self.clusters = nn.Parameter(initial_weights)

    def forward(self, x):
        """ student t-distribution, as same as used in t-SNE algorithm.
            q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            x: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        # print("ID(X) BEFRE UNSQUEEZE",id(x))
        x = (torch.sum(torch.square(torch.unsqueeze(x, dim=1) - self.clusters), dim=2) / self.alpha)
        # print("AFTER UNSQUEEZE", id(x))
        q = 1.0 / (1.0 + x)
        q = torch.pow(q, ((self.alpha + 1.0) / 2.0))
        q = (q.T / torch.sum(q, dim=1)).T
        return q


class scDeepCluster(ZINBAutoEncoder):

    def __init__(
            self,
            input_d,
            n_clusters: int,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            activation='relu',
            weight_initializer='xavier',
            noise_sd: float = 0.,
            **kwargs
    ):
        super(scDeepCluster, self).__init__(input_d, encoder_layers_dim, decoder_layers_dim, latent_layer_out_dim,
                                            batchnorm=False, activation=activation,
                                            weight_initializer=weight_initializer, **kwargs)
        self.input_dim = input_d.shape[1]
        self.n_clusters = n_clusters
        self.pretrain = False
        self.noise_sd = noise_sd
        self.clustering_layer = None

    def pretrain_over(self, initial_cluster_weights):
        self.pretrain = True
        self.clustering_layer = ClusteringLayer(self.input_dim, self.n_clusters, initial_weights=initial_cluster_weights)

    def forward(self, x, size_factors):
        if not self.pretrain:
            x = x + (self.noise_sd ** 0.5)*torch.randn(size=x.shape)
            for encoder_module in self.encode:
                x = encoder_module.linear(x)
                x = x + (self.noise_sd ** 0.5)*torch.randn(size=x.shape)
                x = encoder_module.act_layer(x)
            latent_output = self.latent_layer(x)
            x = self.decode(latent_output)
            mean = self.output_layer(x)
            mean = mean * size_factors.reshape((-1, 1))
            theta = self.theta_layer(x)
            pi = self.pi_layer(x)
            return latent_output, [mean, theta, pi]
        else:
            encoded = self.encode(x)
            latent_output = self.latent_layer(encoded)
            latent_clustering = self.latent_layer(encoded)
            decoded = self.decode(latent_output)
            clustering_output = self.clustering_layer(latent_clustering)
            mean = self.output_layer(decoded)
            mean = mean * size_factors.reshape((-1, 1))
            pi = self.pi_layer(decoded)
            theta = self.theta_layer(decoded)
            return clustering_output, [mean, theta, pi]


class scDeepClusterTrainer(Trainer):

    def __init__(
            self,
            model: scDeepCluster,
            gene_dataset: GeneExpressionDataset,
            use_mask: bool = True,
            scale_factor=1.0,
            ridge_lambda=0.0,
            l1_coeff=0.,
            l2_coeff=0.,
            **kwargs
    ):
        super(scDeepClusterTrainer, self).__init__(model, gene_dataset, **kwargs)

        self.use_mask = use_mask
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.ridge_lambda = ridge_lambda
        self.update_interval = None
        self.tol = None
        self.loss_weights = None
        self.latent_output = None

        (train_set, test_set, val_set) = self.train_test_validation()
        self.register_posterior(train_set, test_set, val_set)

    def pretrain(self, n_epochs=200, lr=0.001, ae_file='ae_weights'):

        self.model.pretrain = False
        self.loss = zinb_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for self.epoch in tqdm(
                range(n_epochs),
                desc="pre-training",
                disable=not self.show_progressbar
        ):
            self.num_iter = 0
            print("Epoch: {}".format(self.epoch + 1))
            for data_tensor in self.data_load_loop("train_set"):
                data, indices = data_tensor

                size_factors_layer = torch.tensor(self.gene_dataset.size_factor[indices, :])
                size_factors_layer = size_factors_layer.cuda() if self.use_cuda else size_factors_layer

                latent_output, output = self.model(data, size_factors_layer)
                raw_data = torch.tensor(self.gene_dataset.raw[indices, :])
                self.current_loss = loss = self.loss(raw_data, output, eps=self.eps, scale_factor=self.scale_factor,
                                                     ridge_lambda=self.ridge_lambda)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.on_iteration_end()

        self.save_checkpoint(ae_file)
        self.model.pretrain = True
        self.model.eval()

    def extract_feature(self, x):  # extract features from before clustering layer
        self.model.eval()
        encoded = self.model.encode(x)
        latent_output = self.model.latent_layer(encoded)
        return latent_output

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        self.model.eval()
        q = self.model.clustering_layer(x)
        return torch.argmax(q, dim=1)

    def target_distribution(self, q):  # target distribution P which enhances the discrimination of soft label Q
        self.model.eval()
        weight = q ** 2 / torch.sum(q, dim=0)
        weight = (weight.T / torch.sum(weight, dim=1)).T
        return weight

    def training_extras_init(self, n_clusters, ae_file, update_interval=20, tol=1e-3, loss_weights=[1, 1]):
        self.update_interval = update_interval
        self.tol = tol
        self.loss_weights = loss_weights
        if not self.model.pretrain and not ae_file:
            print('...pretraining model with default hyperparamters')
            self.pretrain(ae_file=ae_file)
        elif ae_file is not None:
            self.load_checkpoint(ae_file)
            print('...autoencoder weights loaded successfully')
        self.epoch = -1

        self.model.eval()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.extract_feature(torch.tensor(self.gene_dataset.data)).detach().numpy())
        self.y_pred_last = np.copy(self.y_pred)
        self.model.pretrain_over(torch.tensor(kmeans.cluster_centers_))
        self.model.train()

    def on_training_begin(self):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.kl_loss = torch.nn.KLDivLoss()
        self.loss = zinb_loss

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def on_epoch_begin(self):
        self.num_iter = 0
        print("Epoch: {}".format(self.epoch + 1))
        self.model.eval()
        if self.epoch % self.update_interval == 0:
            self.q, _ = self.model(torch.tensor(self.gene_dataset.data), torch.tensor(self.gene_dataset.size_factor))
            self.p = self.target_distribution(self.q).detach()

            self.y_pred = torch.argmax(self.q, dim=1)
            self.y_pred = self.y_pred.detach().numpy()
            # check accuracy of algorithm so far
            if self.gene_dataset.labels is not None:
                acc = np.round(cluster_acc(self.gene_dataset.labels, self.y_pred), 5)
                acc = np.round(adjusted_rand_score(self.gene_dataset.labels, self.y_pred), 5)
                print("Iter: {}\nACC: {:.4f}\n Loss: {:.4f}\n\n".format(self.epoch, acc, self.current_loss))

            # check stop criterion
            delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / self.y_pred.shape[0]
            self.y_pred_last = np.copy(self.y_pred)
            if self.epoch > 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print('Reached tolerance threshold. Stopping training.')
                # todo figure out how to stop training
        self.model.train()

    def on_training_loop(self, data_tensor):
        # print("training loop aaya")
        clustering_output, output = self.model_output(data_tensor)
        data, indices = data_tensor
        raw_data = torch.tensor(self.gene_dataset.raw[indices, :])
        self.optimizer.zero_grad()
        loss1 = self.loss(raw_data, output, eps=self.eps, scale_factor=self.scale_factor, ridge_lambda=self.ridge_lambda)
        # print("CLUSTERING OUTPUT SIZE: ", clustering_output.shape)
        # print("TARGET DIST SIZE: ", self.p.shape)
        # print("CLUSTERING OUTPUT: ", clustering_output)
        # print("TARGET DIST: ", self.p[indices, :])
        loss2 = self.kl_loss(clustering_output.log(), self.p[indices, :])
        # print("LOSS KLDIV: ", loss2)
        loss = self.current_loss = loss1 + loss2

        loss.backward()
        self.optimizer.step()
        # for name, param in self.model.named_parameters():
        #     if name == "clustering_layer.clusters":
        #         print(param.grad)
        #         print(param.grad.shape)
        #         print(param)

    def model_output(self, data_tensor):
        data, indices = data_tensor
        size_factors_layer = torch.tensor(self.gene_dataset.size_factor[indices, :])
        size_factors_layer = size_factors_layer.cuda() if self.use_cuda else size_factors_layer

        clustering_output, output = self.model(data, size_factors_layer)
        return clustering_output, output

    def on_epoch_end(self):
        pass




