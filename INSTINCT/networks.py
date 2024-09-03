import torch
import torch.nn as nn
import torch.nn.functional as F


class GATlayer(nn.Module):

    def __init__(self, input_dim, output_dim, is_last=False):
        super().__init__()

        self.is_last = is_last

        # self.linear = nn.Linear(input_dim, output_dim)
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.v0 = nn.Parameter(torch.Tensor(output_dim, 1))
        self.v1 = nn.Parameter(torch.Tensor(output_dim, 1))

        # nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.v0)
        nn.init.xavier_uniform_(self.v1)

        self.alphas = None

    def forward(self, node_features, graph, tied_alphas=None):

        # X = self.linear(node_features)
        X = torch.matmul(node_features, self.W)

        if not self.is_last:
            # compute attention coefficients
            if tied_alphas is not None:
                alphas = tied_alphas
            else:
                f0 = torch.matmul(X, self.v0)
                f1 = torch.matmul(X, self.v1)
                E = graph * (f0 + f1.T)
                alphas = (torch.sigmoid(E) - 0.5).to_sparse()
                alphas = torch.sparse.softmax(alphas, dim=1)
                alphas = alphas.to_dense()
                self.alphas = alphas

            X = torch.matmul(alphas, X)
            X = F.elu(X, alpha=1.0)

        return X


class GATEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = GATlayer(input_dim, hidden_dims[0])
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = GATlayer(hidden_dims[i-1], hidden_dims[i])
                    self.layer_list.append(layer)

            layer = GATlayer(hidden_dims[-1], latent_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = GATlayer(input_dim, latent_dim, is_last=True)
            self.layer_list.append(layer)

    def forward(self, node_features, graph):

        Z = self.layer_list[0](node_features, graph)

        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                Z = self.layer_list[i](Z, graph)

        return Z


class Discriminator(nn.Module):

    def __init__(self, input_dim, n_cls, hidden_dim_list):
        super().__init__()

        self.num_layer = len(hidden_dim_list) + 1

        if self.num_layer != 1:
            self.linear_list = nn.ModuleList()

            linear = nn.Linear(input_dim, hidden_dim_list[0])
            nn.init.xavier_uniform_(linear.weight)
            self.linear_list.append(linear)

            self.src_layer = nn.Linear(hidden_dim_list[-1], 1)
            nn.init.xavier_uniform_(self.src_layer.weight)

            self.cls_layer = nn.Linear(hidden_dim_list[-1], n_cls)
            nn.init.xavier_uniform_(self.cls_layer.weight)

        else:
            self.src_layer = nn.Linear(input_dim, 1)
            nn.init.xavier_uniform_(self.src_layer.weight)

            self.cls_layer = nn.Linear(input_dim, n_cls)
            nn.init.xavier_uniform_(self.cls_layer.weight)

        if self.num_layer >= 3:
            for i in range(1, len(hidden_dim_list)):
                linear = nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i])
                nn.init.xavier_uniform_(linear.weight)
                self.linear_list.append(linear)

    def forward(self, X):
        if self.num_layer != 1:
            h = self.linear_list[0](X)
            h = torch.softmax(h, dim=1)

            if self.num_layer >= 3:
                for i in range(1, self.num_layer - 1):
                    h = self.linear_list[i](h)
                    h = torch.softmax(h, dim=1)

            src = self.src_layer(h)

            cls = self.cls_layer(h)

        else:
            src = self.src_layer(X)

            cls = self.cls_layer(X)

        return src, cls


class MLPSingleLayer(nn.Module):

    def __init__(self, input_dim, output_dim, is_last=False):
        super().__init__()

        self.is_last = is_last

        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, node_features):

        X = self.linear(node_features)

        if not self.is_last:

            X = F.elu(X, alpha=1.0)

        return X


class MLPEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = MLPSingleLayer(input_dim, hidden_dims[0])
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = MLPSingleLayer(hidden_dims[i - 1], hidden_dims[i])
                    self.layer_list.append(layer)

            layer = MLPSingleLayer(hidden_dims[-1], latent_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = MLPSingleLayer(input_dim, latent_dim, is_last=True)
            self.layer_list.append(layer)

    def forward(self, node_features):

        Z = self.layer_list[0](node_features)

        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                Z = self.layer_list[i](Z)

        return Z


class MLPDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = MLPSingleLayer(latent_dim, hidden_dims[0])
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = MLPSingleLayer(hidden_dims[i - 1], hidden_dims[i])
                    self.layer_list.append(layer)

            layer = MLPSingleLayer(hidden_dims[-1], output_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = MLPSingleLayer(latent_dim, output_dim, is_last=True)
            self.layer_list.append(layer)

    def forward(self, node_features):

        Y = self.layer_list[0](node_features)

        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                Y = self.layer_list[i](Y)

        return Y


class NoiseGenerator(nn.Module):

    def __init__(self, latent_dim, n_cls):
        super().__init__()

        self.linear = nn.Linear(latent_dim + n_cls, latent_dim)

        nn.init.zeros_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)

        # nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, latent_features, node_cls):

        b = torch.concat([latent_features, node_cls], dim=1)

        b = self.linear(b)

        return b






