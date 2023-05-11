# MachineLearningAssignment2

A project for (Computational) Machine Learning

## Project structure
```bash
.
├── data                                 # Data directory
│   ├── raw                             # Raw data directory
│   │   ├── patch_images                # Directory for image patches
│   │   ├── data_labels_extraData.csv   # CSV file for additional data labels
│   │   └── data_labels_mainData.csv    # CSV file for main data labels
│   ├── processed                       # Processed data directory
│   │   ├── cancerous                   # Directory for processed cancerous data
│   │   └── cell_type                   # Directory for processed cell type data
├── models                               # Models directory
│   ├── cancerous                      # Directory for cancerous models
│   └── cell_type                      # Directory for cell type models
├── notebooks                           # Notebooks directory
│   ├── data_exploration.ipynb         # Jupyter notebook for data exploration
│   ├── model_cancerous.ipynb          # Jupyter notebook for cancerous model development
│   ├── model_cell_type.ipynb          # Jupyter notebook for cell type model development
│   └── model_comparison.ipynb         # Jupyter notebook for comparing cancerous and cell type models
├── report                              # Report directory
│   └── report                         # Directory for generated reports
├── src                                 # Source code directory
│   ├── data_utils.ipynb               # Jupyter notebook for data utilities
│   ├── evaluate.py                    # Script for evaluating models
│   ├── model_utils.py                 # Script for model utilities
│   ├── train_cancerous.py             # Script for training cancerous models
│   └── train_cell_type.py             # Script for training cell type models
├── README.md                           # Readme file
└── requirements.txt                    # Requirements file


```