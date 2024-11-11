# Bee Analysis

A comprehensive Python project for analyzing bee behavior in videos using state-of-the-art machine learning techniques. This project includes scripts for data preparation, model training, and video annotation to classify various bee behaviors. While initially designed for bee analysis, the framework is adaptable and can be applied to other datasets.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Creating a Dataset](#creating-a-dataset)
    - [Generating a Dataset CSV](#generating-a-dataset-csv)
    - [Processing the CSV into Training and Testing Data](#processing-the-csv-into-training-and-testing-data)
        - [Synthetic Roach Data](#synthetic-roach-data)
        - [N-Fold Cross Validation](#n-fold-cross-validation)
- [Training a Model](#training-a-model)
- [Annotating Videos](#annotating-videos)
- [Scaling with Slurm](#scaling-with-slurm)
- [Contributing](#contributing)
- [License](#license)

## Features

- Data preparation scripts for generating datasets from raw videos
- Machine learning model training using PyTorch
- Video annotation tools for visualizing predictions and feature intensities
- Support for n-fold cross-validation
- Integration with Slurm for distributed computing

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/bee_analysis.git
cd bee_analysis
```

### Dependencies

Install the dependencies using `pip`:

```bash
pip3 install torch torchvision webdataset ffmpeg
```

Refer to [PyTorch's official guide](https://pytorch.org/get-started/locally/) for additional installation instructions.

## Creating a Dataset

### Generating a Dataset CSV

Run `make_train_csv.sh` to create a CSV file with labels for each video:

```bash
bash make_train_csv.sh /path/to/videos > dataset.csv
```

Ensure the directory contains:

- `logNeg.txt`
- `logNo.txt`
- `logPos.txt`

Each file should list event start timestamps in `YYYYMMDD_HHMMSS` format. These events can represent any classes since the training code is class-agnostic.

### Processing the CSV into Training and Testing Data

Use `VidActRecDataprep.py` to process the CSV:

```bash
python3 VidActRecDataprep.py --width 400 --height 400 --samples 500 --crop_noise 20 \
                             --out_channels 1 --frames_per_sample 1 dataset.csv dataset.tar
```

To scale videos:

```bash
python3 VidActRecDataprep.py --width 200 --height 200 --scale 0.5 --samples 500 \
                             --crop_noise 20 --out_channels 1 --frames_per_sample 1 dataset.csv dataset.tar
```

**Parameters:**

- `--width`, `--height`: Output dimensions
- `--scale`: Scaling factor
- `--samples`: Number of samples per video
- `--crop_noise`: Randomness in cropping
- `--out_channels`: Output channels (e.g., 1 for grayscale)
- `--frames_per_sample`: Frames per sample

Adjust `--crop_x_offset` and `--crop_y_offset` to shift the crop location.

#### Synthetic Roach Data

For roach data, use `roach_csv.py` instead of `make_train_csv.sh`.

#### N-Fold Cross Validation

Split your dataset into `n` chunks and run `VidActRecDataprep.py` for each to perform n-fold cross-validation. This helps in assessing data consistency and model robustness.

## Training a Model

Train the model with `VidActRecTrain.py`:

```bash
python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar train.tar
```

For cross-validation:

```bash
python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar fold1.tar fold2.tar fold3.tar
```

Use `--outname` to specify the model checkpoint filename.

## Annotating Videos

Visualize predictions using `VidActRecAnnotate.py`:

```bash
python3 VidActRecAnnotate.py --datalist dataset.csv \
                             --resume_from model.checkpoint \
                             --modeltype alexnet
```

Options:

- `--class_names`: Set class names
- `--label_classes`: Number of predicted classes

## Scaling with Slurm

For large-scale tasks, integrate with Slurm:

```bash
python3 make_validation_training.py dataset.csv
```

Submit jobs:

```bash
sbatch train_job.slurm
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For significant changes, consider opening an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
