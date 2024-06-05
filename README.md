# INSTINCT
Multi-sample integration of spatial chromatin accessibility sequencing data via stochastic domain translation
![Overview_of_INSTINCT](https://github.com/yyLIU12138/INSTINCT/assets/130898915/de84d937-361b-4083-af29-8b5ea03b58ec)

## System requirements
The package development version is tested on Windows operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 20.04  
Windows


## Installation
Clone the repository. 

```
git clone https://github.com/yyLIU12138/INSTINCT.git
cd INSTINCT
```

Create an environment.

```
conda create -n epi_INSTINCT python=3.10
conda activate epi_INSTINCT
```

Install the required packages.

```
pip install -r requirement.txt
```

Install INSTINCT.

```
python setup.py build
python setup.py install
```

Installation takes a few minutes.


## Tutorial

Import the package.
```
import torch
import anndata as ad
from sklearn.decomposition import PCA
import INSTINCT
import warnings
warnings.filterwarnings("ignore")
```

Load the anndata type data samples into a list.

```
data_dir = '../../data/spMOdata/EpiTran_MouseBrain_Jiang2023/preprocessed/'
slice_name_list = ['E11_0-S1', 'E13_5-S1', 'E15_5-S1', 'E18_5-S1']
cas_list = [ad.read_h5ad(data_dir + sample + '_atac.h5ad') for sample in slice_name_list]
for j in range(len(cas_list)):
    cas_list[j].obs_names = [x + '_' + slice_name_list[j] for x in cas_list[j].obs_names]
```

Merge the peaks.

```
cas_list = INSTINCT.peak_sets_alignment(cas_list)
```

Preprocessing (If the data samples already incorparate fragment count matrices, then set use_fragment_count=False).

```
adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list)
INSTINCT.preprocess_CAS(cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03)
```

Use PCA to reduce the dimensionality of the concatenated data to 100. The matrix of shape N*100 should be stored in adata_concat.obsm['X_pca'].

```
pca = PCA(n_components=100, random_state=1234)
input_matrix = pca.fit_transform(adata_concat.X.toarray())
adata_concat.obsm['X_pca'] = input_matrix
```

Construct the neighbor graph

```
INSTINCT.create_neighbor_graph(cas_list, adata_concat)
```

Train the model. 
The low-dimensional representations for spots are stored in .obsm['INSTINCT_latent'] of each slice in cas_list.

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INSTINCT_model = INSTINCT.INSTINCT_Model(cas_list,
                                         adata_concat,
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
                                         early_stop=False,  # use the latent loss to control the number of training steps
                                         min_steps=500,  # the least number of steps when training the whole model
                                         use_cos=True,  # use cosine similarity to find the nearest neighbors
                                         margin=10,  # the margin of latent loss
                                         alpha=1,  # the hyperparameter for triplet loss
                                         k=50,  # the amount of neighbors to find
                                         device=device)

INSTINCT_model.train(report_loss=True, report_interval=100)

INSTINCT_model.eval(cas_list)
```

Training the model takes about one minute using GPU (RTX 4090D 24GB).

