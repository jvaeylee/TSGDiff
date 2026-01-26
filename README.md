# TSGDiff
## Directory Structure & File Description
| File/Directory            | Description                                                                                                                               |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `config.py`               | Global configuration file: Defines hyperparameters (e.g., learning rate, batch size, training epochs), path configurations, environment variables, etc., for the entire process of model training, data loading, and evaluation |
| `evaluation_utils.py`     | Evaluation utility class: Encapsulates functions related to model performance evaluation (e.g., metric calculation, result verification, test set evaluation logic) to support model effect verification during training/testing phases |
| `data_utils.py`           | Data utility class: Implements dataset loading, preprocessing (e.g., data cleaning, format conversion, normalization), data augmentation, dataset splitting, and iterator construction |
| `main.py`                 | Project entry file: Integrates all modules and defines the core processes of training/testing/inference (e.g., parameter parsing, model initialization, data loading, training loop, result saving) |
| `graph_metric.py`         | Graph metric（Topo-FID） calculation file: Encapsulates the calculation logic of dedicated evaluation metrics for graph-structured data                                                                 |
| `model.py`                | Model definition file: Implements the core architecture of the TSGDiff model (e.g., network layers, differential modules), including model classes, forward propagation logic, and submodule definitions |
| `requirements.txt`        | Dependency list: Lists all Python packages and their corresponding versions (e.g., torch, numpy, pandas) required to run the project, supporting one-click installation of dependencies |
| `visualization_utils.py`  | Visualization utility class: Provides functions for visualizing experimental results                                                                 |
| `train_utils.py`          | Training utility class: Encapsulates general functions for the training process (e.g., optimizer initialization, learning rate scheduling, training step loop, model saving/loading, early stopping logic) |
| `config/`                 | Configuration subdirectory: Stores configuration files (e.g., exclusive configuration yaml/json files for different datasets/experiments) to supplement the flexible configuration needs of the global config.py |
| `datasets/`               | Dataset directory: Stores raw datasets, preprocessed dataset files, or index/annotation files related to datasets                                                                 |
