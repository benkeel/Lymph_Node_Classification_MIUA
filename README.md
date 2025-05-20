#Lymph_Node_Classification_MIUA
Official implementation of MIUA paper: Interpretable prediction of lymph node metastasis in rectal cancer MRI using variational autoencoders

## To initalise the environment:
pip install -r requirements.txt



## Repo Structure

```
.
├── Data_Preprocessing
│   ├── Data_Inspect.ipynb
│   ├── Image_Data_Extraction.ipynb
│   └── Preprocess_Dataset.ipynb
│
├── Results
│   └── error_analysis.png
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
├── README.md
└── requirements.txt
```

