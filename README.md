# [AAAI 2026] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

## 🔧 Installation

See [requirements.txt](requirements.txt) for details.

## 📊 Dataset

The model is designed for time series generation, and the data set used is shown in the [datasets](datasets) directory.

## ⚙️ Configuration

### Parameter configuration

| Argument               | Type  | Default          | Description                                                                 |
| ---------------------- | ----- | ---------------- | --------------------------------------------------------------------------- |
| `--data_file`          | str   | datasets/{filename}.csv         | Path to input data file (ETTh.csv)                                          |
| `--output_dir`         | str   | output           | Directory to save output results (including visualizations and metrics)     |
| `--slide_win`          | int   | 48               | Sliding window size for time series slicing                                 |
| `--slide_stride`       | int   | 1                | Sliding stride for time series slicing                                      |
| `--hidden_dim`         | int   | 1600             | Graph encoder hidden layer dimension                                        |
| `--embed_dim`          | int   | 1600             | Graph encoder embedding dimension                                           |
| `--use_period_index`   | int   | 2                | Index of period used for graph edge generation                              |
| `--batch_size`         | int   | 128              | Batch size for training                                                     |
| `--epochs`             | int   | 500              | Number of training epochs                                                   |
| `--learning_rate`      | float | 0.01             | Learning rate for optimizer                                                 |
| `--normalize_data`     | bool  | True             | Whether to normalize input data                                             |
| `--feature_range`      | tuple | (-1, 1)          | Feature range for Min-Max normalization                                     |
| `--kl_weight`          | float | 0.2              | Weight for KL divergence loss in total loss calculation                     |
| `--diffusion_weight`   | float | 1.0              | Weight for Diffusion model loss in total loss calculation                   |
| `--num_timesteps`      | int   | 1000             | Number of timesteps for Diffusion model                                     |
| `--beta_start`         | float | 0.0001           | Starting value of beta for Diffusion model noise schedule                   |
| `--beta_end`           | float | 0.02             | Ending value of beta for Diffusion model noise schedule                     |
| `--nblocks`            | int   | 3                | Number of intermediate blocks in Diffusion Block                            |
| `--nunits`             | int   | 64               | Number of units per Diffusion Block                                         |
| `--k`                  | int   | 2                | Multiplier for extending edges with remaining periods (graph construction)  |
| `--ff_weight`          | float | sqrt(slide_win)/5| Weight for Fourier loss (scaled by sliding window size)                     |

## 📊 Results

Results are automatically saved to:

- **Original data**: `output/original_data.npy`
- **Generated data**: `output/generated_data.npy`
- **Visualizations**: `output/visualizations/`
- **Evaluation metrics**: `output/metrics.txt`

## 📁 Project Structure

```
TSGDiff/
├── main.py                          # Main entry point for training/inference/evaluation
├── model.py                         # Core implementation of TSGDiff Model
├── config.py                        # Global hyperparameter & path configurations
├── train_utils.py                   # Training utilities
├── data_utils.py                    # Data loading & preprocessing for temporal-spatial graph
├── graph_metric.py                  # Topo-FID calculations
├── evaluation_utils.py              # Evaluation metrics
├── visualization_utils.py           # Visualization tools
├── requirements.txt                 # Python dependencies
├── config/
│   └── *.yaml                       # Scene/dataset-specific configuration files
└── datasets/
    └── *.csv                        # Temporal-spatial graph datasets
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{TSGDiff,
  title={TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective},
  author={Shen, Lifeng and Li, Xuyang and Long, Lele},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please contact:

- Xuyang Li : lixuyang.lee@foxmail.com
