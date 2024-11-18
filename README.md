# VAE-MLP_CVPR
Official implementation of CVPR paper: Interpretable prediction of lymph node metastasis using variational autoencoders on rectal cancer MRI for patients without neo-adjuvant treatment

### To run this code with your own data you can get the environment using:
pip install -r requirements.txt




## Repo Structure

```
.
├── Data_Preprocessing
│   ├── Data_Inspect.ipynb
│   ├── Image_Data_Extraction.ipynb
│   └── Preprocess_Dataset.ipynb
│
├── VAE-MLP
│   └── models
│       ├── MLP_model.py
│       └── VAE_2D_model.py
│
│   └── scripts
│       ├── Ablation_study.ipynb
│       ├── CNN_MLP_hyperparam_sweep.ipynb
│       ├── Grad-CAM.ipynb
│       ├── Latent_clustering.iypnb
│       ├── Latent_traversals.iypnb
│       ├── MLP_hyperparam_sweep.ipynb
│       ├── VAE_MLP_Cross_Validation.ipynb
│       ├── VAE_hyperparam_sweep.ipynb
│       ├── stats.ipynb
│       └── visualise_augmentations.ipynb
│
│   └── utils
│       ├── datasets.py
│       ├── loss_functions.py
│       ├── train_and_test_functions.py
│       └── utility_code.py
│
├── requirements.txt
└── README.md
```

