import csv
import os
import time
import random
import numpy as np
import torch.optim as optim

from .networks import *
from .loss_functions import *


class INSTINCT_Model():

    def __init__(self,
                 adata_list,  # slice list
                 adata_concat,  # the concatenated data
                 input_mat_key='X_pca',  # the key of the input matrix in adata_concat.obsm
                 input_dim=100,  # the input dimension
                 hidden_dims_G=[50],  # hidden dimensions of the encoder and the decoder
                 latent_dim=30,  # the dimension of latent space
                 hidden_dims_D=[50],  # hidden dimensions of the discriminator
                 lambda_adv=1,  # hyperparameter for the adversarial loss
                 lambda_cls=10,  # hyperparameter for the classification loss
                 lambda_la=20,  # hyperparameter for the latent loss
                 lambda_rec=10,  # hyperparameter for the reconstruction loss
                 seed=1236,  # random seed
                 learn_rates=[1e-3, 5e-4],  # learning rate
                 training_steps=[500, 500],  # training_steps
                 use_cos=True,  # use cosine similarity to find the nearest neighbors, use euclidean distance otherwise
                 margin=10,  # the margin of latent loss
                 alpha=1,  # the hyperparameter for triplet loss
                 k=50,  # the amount of neighbors to find
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):

        self.lambda_adv = lambda_adv
        self.lambda_cls = lambda_cls
        self.lambda_la = lambda_la
        self.lambda_rec = lambda_rec

        self.latent_dim = latent_dim
        self.use_cos = use_cos
        self.margin = margin
        self.alpha = alpha
        self.k = k

        self.device = device

        self.n_cls = len(adata_list)
        self.n_spots = adata_concat.shape[0]
        self.training_steps = training_steps

        self.slice_index = np.zeros((self.n_spots, ))
        self.indices = [0]
        idx = 0
        for i in range(self.n_cls):
            idx += adata_list[i].shape[0]
            self.indices.append(idx)
            self.slice_index[self.indices[-2]:self.indices[-1]] = i

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        # networks
        self.encoder = GATEncoder(input_dim, hidden_dims_G, latent_dim).to(self.device)
        self.decoder = MLPDecoder(latent_dim, hidden_dims_G, input_dim).to(self.device)
        self.noise_generator = NoiseGenerator(latent_dim, self.n_cls).to(self.device)
        self.discriminator = Discriminator(input_dim, self.n_cls, hidden_dims_D).to(self.device)

        # parameters
        self.params_G = list(self.encoder.parameters()) + \
                        list(self.decoder.parameters()) + \
                        list(self.noise_generator.parameters())
        self.params_D = self.discriminator.parameters()

        # optimizers
        self.optimizer_G = optim.Adam(self.params_G, lr=learn_rates[0])
        self.optimizer_D = optim.Adam(self.params_D, lr=learn_rates[1])

        # data
        self.X = torch.from_numpy(adata_concat.obsm[input_mat_key]).float().to(self.device)
        self.X_target = torch.tensor(self.slice_index, dtype=torch.int64).to(self.device)
        self.X_target_matrix = torch.eye(self.n_cls)[self.slice_index.astype(float)].to(self.device)

        if 'graph_list' in adata_concat.uns.keys():
            self.G = np.zeros((adata_concat.X.shape[0], adata_concat.X.shape[0]), dtype=float)
            for i in range(len(adata_list)):
                subgraph = adata_concat.uns['graph_list'][i].copy().astype(float)
                self.G[self.indices[i]:self.indices[i + 1], self.indices[i]:self.indices[i + 1]] = subgraph
            self.G = torch.from_numpy(self.G).float().to(self.device)
        elif 'graph' in adata_concat.obsm.keys():
            self.G = torch.from_numpy(adata_concat.obsm['graph'].copy()).float().to(self.device)
        else:
            raise ValueError('No graph is provided !')

    def train(self, report_loss=True, report_interval=100):

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.encoder.train()
        self.decoder.train()
        self.noise_generator.train()
        self.discriminator.train()

        if self.training_steps[0]:
            print('Start training the autoencoder and the discriminator ...\n')

        for step in range(self.training_steps[0]):

            Z_X = self.encoder(self.X, self.G)
            X_rec = self.decoder(Z_X)

            X_src, X_cls = self.discriminator(self.X)
            X_rec_src, X_rec_cls = self.discriminator(X_rec)

            loss_adv_g = adversarial_loss_g(X_rec_src)
            loss_adv_d = adversarial_loss_d(X_src, X_rec_src)

            loss_rec = mse_loss(self.X, X_rec)

            loss_cls_real = classification_loss(X_cls, self.X_target)
            loss_cls_fake = classification_loss(X_rec_cls, self.X_target)

            # D
            loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
            self.optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)

            # G
            loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + self.lambda_rec * loss_rec
            self.optimizer_G.zero_grad()
            loss_g.backward()

            self.optimizer_D.step()
            self.optimizer_G.step()

            if report_loss:
                if (step + 1) % report_interval == 0 or step == 0:
                    print("Step: %s" % step)
                    print("loss_G: %.4f, loss_D: %.4f" % (loss_g.item(), loss_d.item()))
                    print("loss_adv_g: %.4f, loss_adv_d: %.4f, loss_rec: %.4f, "
                          "loss_cls_real: %.4f, loss_cls_fake: %.4f"
                          % (loss_adv_g.item(), loss_adv_d.item(), loss_rec.item(),
                             loss_cls_real.item(), loss_cls_fake.item()))

                    _, X_pred = torch.max(X_cls, dim=1)
                    X_correct = torch.sum(X_pred == self.X_target).item() / self.n_spots
                    _, X_rec_pred = torch.max(X_rec_cls, dim=1)
                    X_rec_correct = torch.sum(X_rec_pred == self.X_target).item() / self.n_spots
                    print(f'X_pred accuracy: {X_correct: .4f}')
                    print(f'X_rec_pred accuracy: {X_rec_correct: .4f}\n')

            torch.cuda.empty_cache()

        if self.training_steps[0]:
            print('Done !\n')

        if self.training_steps[1]:
            print('Start training the whole model ...\n')

        for step in range(self.training_steps[1]):

            # generate transferred cls
            # cls_for_slices = [random.randint(0, self.n_cls - 1) for _ in range(self.n_cls)]
            cls_for_slices = [random.choice(random.sample([i for i in range(self.n_cls) if i != j], self.n_cls - 1))
                              for j in range(self.n_cls)]
            Y_target = np.zeros((self.n_spots,))
            for i in range(self.n_cls):
                Y_target[self.indices[i]:self.indices[i + 1]] = cls_for_slices[i]
            Y_target = torch.tensor(Y_target, dtype=torch.int64).to(self.device)

            Z_X = self.encoder(self.X, self.G)

            Noise_X = self.noise_generator(Z_X, self.X_target_matrix)
            Z_X_hat = Z_X + Noise_X
            X_rec = self.decoder(Z_X_hat)  # without_D_NG: replace Z_X_hat with Z_X, use loss_rec and loss_la only

            with torch.no_grad():
                knn_indices_list, random_indices_list = self.find_knn(self.indices, cls_for_slices,
                                                                      Z_X, self.use_cos, self.k)

            Noise_Y = self.simulate_noise(self.indices, cls_for_slices, Noise_X, knn_indices_list)
            Z_Y_hat = Z_X + Noise_Y
            Y = self.decoder(Z_Y_hat)

            D_concat = self.calculate_distance(self.indices, cls_for_slices, Z_X, knn_indices_list, random_indices_list)

            X_src, X_cls = self.discriminator(self.X)
            Y_src, Y_cls = self.discriminator(Y)

            loss_adv_g = adversarial_loss_g(Y_src)
            loss_adv_d = adversarial_loss_d(X_src, Y_src)

            loss_la = latent_loss(D_concat, self.alpha, margin=self.margin)
            loss_rec = mse_loss(self.X, X_rec)

            loss_cls_real = classification_loss(X_cls, self.X_target)
            loss_cls_fake = classification_loss(Y_cls, Y_target)

            # D
            loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
            self.optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)

            # G
            loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + \
                     self.lambda_la * loss_la + self.lambda_rec * loss_rec
            self.optimizer_G.zero_grad()
            loss_g.backward()

            self.optimizer_D.step()
            self.optimizer_G.step()

            if report_loss:
                if (step + 1) % report_interval == 0 or step == 0:
                    print("Step: %s" % step)
                    print("loss_G: %.4f, loss_D: %.4f" % (loss_g.item(), loss_d.item()))
                    print("loss_adv_g: %.4f, loss_adv_d: %.4f, loss_la: %.4f, "
                          "loss_rec: %.4f, loss_cls_real: %.4f, loss_cls_fake: %.4f"
                          % (loss_adv_g.item(), loss_adv_d.item(), loss_la.item(),
                             loss_rec.item(), loss_cls_real.item(), loss_cls_fake.item()))

                    _, X_pred = torch.max(X_cls, dim=1)
                    X_correct = torch.sum(X_pred == self.X_target).item() / self.n_spots
                    _, Y_pred = torch.max(Y_cls, dim=1)
                    Y_correct = torch.sum(Y_pred == Y_target).item() / self.n_spots
                    print(f'X_pred accuracy: {X_correct: .4f}')
                    print(f'Y_pred accuracy: {Y_correct: .4f}\n')

            torch.cuda.empty_cache()

        if self.training_steps[1]:
            print('Done !\n')

        end_time = time.time()

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        eval_time = end_time - begin_time
        print("Takes %.2f seconds in total\n" % eval_time)

    def eval(self, adata_list, slice_names=None, save_dir_data='../../results/', save=False):

        self.encoder.eval()
        self.decoder.eval()
        self.noise_generator.eval()
        self.discriminator.eval()

        Z_X = self.encoder(self.X, self.G)
        Noise_X = self.noise_generator(Z_X, self.X_target_matrix)

        Z_X = Z_X.cpu().detach().numpy()
        Noise_X = Noise_X.cpu().detach().numpy()

        for i in range(len(adata_list)):
            adata_list[i].obsm['INSTINCT_latent'] = Z_X[self.indices[i]:self.indices[i + 1]]
            adata_list[i].obsm['INSTINCT_latent_noise'] = Noise_X[self.indices[i]:self.indices[i + 1]]

            if save:
                if not os.path.exists(save_dir_data):
                    os.makedirs(save_dir_data)
                adata_list[i].write_h5ad(save_dir_data + 'INSTINCT_' + slice_names[i] + f'_{self.training_steps}.h5ad')

    def find_knn(self, spot_idx_list, target_slice_list, latent_features, use_cos=True, k=50):

        knn_indices_list = []
        random_indices_list = []

        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            if use_cos:
                _, knn_indices = F.cosine_similarity(latent_features[start_idx:end_idx].unsqueeze(1),
                                                     latent_features[target_start_idx:target_end_idx].unsqueeze(0),
                                                     dim=2).topk(k, dim=1, largest=True)

            else:
                _, knn_indices = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                                            latent_features[target_start_idx:target_end_idx].unsqueeze(0),
                                            p=2, dim=2).topk(k, dim=1, largest=False)

            knn_indices_list.append(knn_indices)

            random_indices = torch.tensor(np.array([random.choices(range(0, end_idx - start_idx))
                                                    for _ in range(k)]).T[0]).long().to(self.device)
            random_indices_list.append(random_indices)

        return knn_indices_list, random_indices_list

    def simulate_noise(self, spot_idx_list, target_slice_list, ori_noise, knn_indices_list):

        new_noise_concat = []

        for i in range(len(target_slice_list)):

            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            selected_rows = ori_noise[target_start_idx:target_end_idx][knn_indices_list[i]]
            new_noise = selected_rows.mean(dim=1)

            new_noise_concat.append(new_noise)

        new_noise_concat = torch.concat(new_noise_concat, dim=0)

        return new_noise_concat

    def calculate_distance(self, spot_idx_list, target_slice_list, latent_features,
                           knn_indices_list, random_indices_list):

        D_concat = []

        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            D_pos = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[target_start_idx:target_end_idx][knn_indices_list[i]].unsqueeze(0),
                               p=2, dim=3).squeeze(0)

            D_neg = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[start_idx:end_idx][random_indices_list[i]].unsqueeze(0),
                               p=2, dim=2)

            D_concat.append(D_pos - D_neg)

        D_concat = torch.concat(D_concat, dim=0)

        return D_concat


class INSTINCT_MLP_Model():

    def __init__(self,
                 adata_list,  # slice list
                 adata_concat,  # the concatenated data
                 input_mat_key='X_pca',  # the key of the input matrix in adata_concat.obsm
                 input_dim=100,  # the input dimension
                 hidden_dims_G=[50],  # hidden dimensions of the encoder and the decoder
                 latent_dim=30,  # the dimension of latent space
                 hidden_dims_D=[50],  # hidden dimensions of the discriminator
                 lambda_adv=1,  # hyperparameter for the adversarial loss
                 lambda_cls=10,  # hyperparameter for the classification loss
                 lambda_la=20,  # hyperparameter for the latent loss
                 lambda_rec=10,  # hyperparameter for the reconstruction loss
                 seed=1236,  # random seed
                 learn_rates=[1e-3, 5e-4],  # learning rate
                 training_steps=[500, 500],  # training_steps
                 use_cos=True,  # use cosine similarity to find the nearest neighbors, use euclidean distance otherwise
                 margin=10,  # the margin of latent loss
                 alpha=1,  # the hyperparameter for triplet loss
                 k=50,  # the amount of neighbors to find
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):

        self.lambda_adv = lambda_adv
        self.lambda_cls = lambda_cls
        self.lambda_la = lambda_la
        self.lambda_rec = lambda_rec

        self.latent_dim = latent_dim
        self.use_cos = use_cos
        self.margin = margin
        self.alpha = alpha
        self.k = k

        self.device = device

        self.n_cls = len(adata_list)
        self.n_spots = adata_concat.shape[0]
        self.training_steps = training_steps

        self.slice_index = np.zeros((self.n_spots, ))
        self.indices = [0]
        idx = 0
        for i in range(self.n_cls):
            idx += adata_list[i].shape[0]
            self.indices.append(idx)
            self.slice_index[self.indices[-2]:self.indices[-1]] = i

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        # networks
        self.encoder = MLPEncoder(input_dim, hidden_dims_G, latent_dim).to(self.device)
        self.decoder = MLPDecoder(latent_dim, hidden_dims_G, input_dim).to(self.device)
        self.noise_generator = NoiseGenerator(latent_dim, self.n_cls).to(self.device)
        self.discriminator = Discriminator(input_dim, self.n_cls, hidden_dims_D).to(self.device)

        # parameters
        self.params_G = list(self.encoder.parameters()) + \
                        list(self.decoder.parameters()) + \
                        list(self.noise_generator.parameters())
        self.params_D = self.discriminator.parameters()

        # optimizers
        self.optimizer_G = optim.Adam(self.params_G, lr=learn_rates[0])
        self.optimizer_D = optim.Adam(self.params_D, lr=learn_rates[1])

        # data
        self.X = torch.from_numpy(adata_concat.obsm[input_mat_key]).float().to(self.device)
        self.X_target = torch.tensor(self.slice_index, dtype=torch.int64).to(self.device)
        self.X_target_matrix = torch.eye(self.n_cls)[self.slice_index.astype(float)].to(self.device)

    def train(self, report_loss=True, report_interval=100):

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.encoder.train()
        self.decoder.train()
        self.noise_generator.train()
        self.discriminator.train()

        if self.training_steps[0]:
            print('Start training the autoencoder and the discriminator ...\n')

        for step in range(self.training_steps[0]):

            Z_X = self.encoder(self.X)
            X_rec = self.decoder(Z_X)

            X_src, X_cls = self.discriminator(self.X)
            X_rec_src, X_rec_cls = self.discriminator(X_rec)

            loss_adv_g = adversarial_loss_g(X_rec_src)
            loss_adv_d = adversarial_loss_d(X_src, X_rec_src)

            loss_rec = mse_loss(self.X, X_rec)

            loss_cls_real = classification_loss(X_cls, self.X_target)
            loss_cls_fake = classification_loss(X_rec_cls, self.X_target)

            # D
            loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
            self.optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)

            # G
            loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + self.lambda_rec * loss_rec
            self.optimizer_G.zero_grad()
            loss_g.backward()

            self.optimizer_D.step()
            self.optimizer_G.step()

            if report_loss:
                if (step + 1) % report_interval == 0 or step == 0:
                    print("Step: %s" % step)
                    print("loss_G: %.4f, loss_D: %.4f" % (loss_g.item(), loss_d.item()))
                    print("loss_adv_g: %.4f, loss_adv_d: %.4f, loss_rec: %.4f, "
                          "loss_cls_real: %.4f, loss_cls_fake: %.4f"
                          % (loss_adv_g.item(), loss_adv_d.item(), loss_rec.item(),
                             loss_cls_real.item(), loss_cls_fake.item()))

                    _, X_pred = torch.max(X_cls, dim=1)
                    X_correct = torch.sum(X_pred == self.X_target).item() / self.n_spots
                    _, X_rec_pred = torch.max(X_rec_cls, dim=1)
                    X_rec_correct = torch.sum(X_rec_pred == self.X_target).item() / self.n_spots
                    print(f'X_pred accuracy: {X_correct: .4f}')
                    print(f'X_rec_pred accuracy: {X_rec_correct: .4f}\n')

            torch.cuda.empty_cache()

        if self.training_steps[0]:
            print('Done !\n')

        if self.training_steps[1]:
            print('Start training the whole model ...\n')

        for step in range(self.training_steps[1]):

            # generate transferred cls
            # cls_for_slices = [random.randint(0, self.n_cls - 1) for _ in range(self.n_cls)]
            cls_for_slices = [random.choice(random.sample([i for i in range(self.n_cls) if i != j], self.n_cls - 1))
                              for j in range(self.n_cls)]
            Y_target = np.zeros((self.n_spots,))
            for i in range(self.n_cls):
                Y_target[self.indices[i]:self.indices[i + 1]] = cls_for_slices[i]
            Y_target = torch.tensor(Y_target, dtype=torch.int64).to(self.device)

            Z_X = self.encoder(self.X)

            Noise_X = self.noise_generator(Z_X, self.X_target_matrix)
            Z_X_hat = Z_X + Noise_X
            X_rec = self.decoder(Z_X_hat)  # without_D_NG: replace Z_X_hat with Z_X, use loss_rec and loss_la only

            with torch.no_grad():
                knn_indices_list, random_indices_list = self.find_knn(self.indices, cls_for_slices,
                                                                      Z_X, self.use_cos, self.k)

            Noise_Y = self.simulate_noise(self.indices, cls_for_slices, Noise_X, knn_indices_list)
            Z_Y_hat = Z_X + Noise_Y
            Y = self.decoder(Z_Y_hat)

            D_concat = self.calculate_distance(self.indices, cls_for_slices, Z_X, knn_indices_list, random_indices_list)

            X_src, X_cls = self.discriminator(self.X)
            Y_src, Y_cls = self.discriminator(Y)

            loss_adv_g = adversarial_loss_g(Y_src)
            loss_adv_d = adversarial_loss_d(X_src, Y_src)

            loss_la = latent_loss(D_concat, self.alpha, margin=self.margin)
            loss_rec = mse_loss(self.X, X_rec)

            loss_cls_real = classification_loss(X_cls, self.X_target)
            loss_cls_fake = classification_loss(Y_cls, Y_target)

            # D
            loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
            self.optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)

            # G
            loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + \
                     self.lambda_la * loss_la + self.lambda_rec * loss_rec
            self.optimizer_G.zero_grad()
            loss_g.backward()

            self.optimizer_D.step()
            self.optimizer_G.step()

            if report_loss:
                if (step + 1) % report_interval == 0 or step == 0:
                    print("Step: %s" % step)
                    print("loss_G: %.4f, loss_D: %.4f" % (loss_g.item(), loss_d.item()))
                    print("loss_adv_g: %.4f, loss_adv_d: %.4f, loss_la: %.4f, "
                          "loss_rec: %.4f, loss_cls_real: %.4f, loss_cls_fake: %.4f"
                          % (loss_adv_g.item(), loss_adv_d.item(), loss_la.item(),
                             loss_rec.item(), loss_cls_real.item(), loss_cls_fake.item()))

                    _, X_pred = torch.max(X_cls, dim=1)
                    X_correct = torch.sum(X_pred == self.X_target).item() / self.n_spots
                    _, Y_pred = torch.max(Y_cls, dim=1)
                    Y_correct = torch.sum(Y_pred == Y_target).item() / self.n_spots
                    print(f'X_pred accuracy: {X_correct: .4f}')
                    print(f'Y_pred accuracy: {Y_correct: .4f}\n')

            torch.cuda.empty_cache()

        if self.training_steps[1]:
            print('Done !\n')

        end_time = time.time()

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        eval_time = end_time - begin_time
        print("Takes %.2f seconds in total\n" % eval_time)

    def eval(self, adata_list, slice_names=None, save_dir_data='../../results/', save=False):

        self.encoder.eval()
        self.decoder.eval()
        self.noise_generator.eval()
        self.discriminator.eval()

        Z_X = self.encoder(self.X)
        Noise_X = self.noise_generator(Z_X, self.X_target_matrix)

        Z_X = Z_X.cpu().detach().numpy()
        Noise_X = Noise_X.cpu().detach().numpy()

        for i in range(len(adata_list)):
            adata_list[i].obsm['INSTINCT_latent'] = Z_X[self.indices[i]:self.indices[i + 1]]
            adata_list[i].obsm['INSTINCT_latent_noise'] = Noise_X[self.indices[i]:self.indices[i + 1]]

            if save:
                if not os.path.exists(save_dir_data):
                    os.makedirs(save_dir_data)
                adata_list[i].write_h5ad(save_dir_data + 'INSTINCT_' + slice_names[i] + f'_{self.training_steps}.h5ad')

    def find_knn(self, spot_idx_list, target_slice_list, latent_features, use_cos=True, k=50):

        knn_indices_list = []
        random_indices_list = []

        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            if use_cos:
                _, knn_indices = F.cosine_similarity(latent_features[start_idx:end_idx].unsqueeze(1),
                                                     latent_features[target_start_idx:target_end_idx].unsqueeze(0),
                                                     dim=2).topk(k, dim=1, largest=True)

            else:
                _, knn_indices = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                                            latent_features[target_start_idx:target_end_idx].unsqueeze(0),
                                            p=2, dim=2).topk(k, dim=1, largest=False)

            knn_indices_list.append(knn_indices)

            random_indices = torch.tensor(np.array([random.choices(range(0, end_idx - start_idx))
                                                    for _ in range(k)]).T[0]).long().to(self.device)
            random_indices_list.append(random_indices)

        return knn_indices_list, random_indices_list

    def simulate_noise(self, spot_idx_list, target_slice_list, ori_noise, knn_indices_list):

        new_noise_concat = []

        for i in range(len(target_slice_list)):

            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            selected_rows = ori_noise[target_start_idx:target_end_idx][knn_indices_list[i]]
            new_noise = selected_rows.mean(dim=1)

            new_noise_concat.append(new_noise)

        new_noise_concat = torch.concat(new_noise_concat, dim=0)

        return new_noise_concat

    def calculate_distance(self, spot_idx_list, target_slice_list, latent_features,
                           knn_indices_list, random_indices_list):

        D_concat = []

        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            D_pos = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[target_start_idx:target_end_idx][knn_indices_list[i]].unsqueeze(0),
                               p=2, dim=3).squeeze(0)

            D_neg = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[start_idx:end_idx][random_indices_list[i]].unsqueeze(0),
                               p=2, dim=2)

            D_concat.append(D_pos - D_neg)

        D_concat = torch.concat(D_concat, dim=0)

        return D_concat



