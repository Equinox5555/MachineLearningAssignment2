# MachineLearningAssignment2

A project for (Computational) Machine Learning

## Project structure
```bash
.
<<<<<<< HEAD
├── data 
│   ├── raw
│   │   ├── patch_images               
│   │   ├── data_labels_extraData.csv     
│   │   └── data_labels_mainData.csv     
│   ├── task_one
│   │   ├── train.csv               
│   │   ├── val.csv     
│   │   └── test.csv   
│   ├── task_two
│   │   ├── train.csv               
│   │   ├── val.csv     
│   │   └── test.csv   
├── task_one     
│   ├── models                          
│   ├── notebooks                       
│   │   ├── i_preprocessing.ipynb       
│   │   ├── ii_eda.ipynb                
│   │   ├── iii_model_training.ipynb    
│   │   ├── iv_model_evaluation.ipynb   
│   │   ├── v_model_finetuning.ipynb    
│   │   └── vi_model_deployment.ipynb   
├── task_two     
│   ├── models                          
│   ├── notebooks                       
│   │   ├── i_preprocessing.ipynb       
│   │   ├── ii_eda.ipynb                
│   │   ├── iii_model_training.ipynb    
│   │   ├── iv_model_evaluation.ipynb   
│   │   ├── v_model_finetuning.ipynb   
│   │   └── vi_model_deployment.ipynb   
├── src                            
├── README.md                       




│   ├── __init__.py                 # make src a Python module
│   ├── config.py                   # store configs 
│   ├── process.py                  # process data before training model
│   ├── run_notebook.py             # run notebook
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py



│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   ├── raw                         # raw data

#├── docs                            # documentation for the project
#├── Makefile                        # store useful commands to set up the environment
=======
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


>>>>>>> 3bda85a32b7de704b8c47852ee036bf284676bff
```