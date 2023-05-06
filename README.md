# MachineLearningAssignment2

A project for (Computational) Machine Learning

## Project structure
```bash
.
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
```