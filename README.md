# MSPLGFormer

PyTorch implementation of **MSPLGFormer** for multivariate time series classification.
The model combines multi-scale patch representations, a lightweight Transformer
with a convolutional feed-forward network (ConvFFN), graph message passing,
pooling, and learnable fusion across scales.

## Architecture

```text
Input [B, C, L]
 -> Temporal Conv2D embedding and channel aggregation
 -> Parallel patch branches: lengths [8, 16], strides [4, 8]
      -> Patch projection
      -> Lightweight Transformer encoder x 1
           -> Pre-norm multi-head self-attention (8 heads)
           -> Pre-norm ConvFFN (hidden dimension 128, temporal kernel 7)
      -> Two parallel MPNN branches
      -> PAP: concatenate mean and max pooling in each branch
      -> Concatenate the two branch outputs
 -> Weighted sum across scales using learnable softmax weights
 -> Linear classifier
 -> Class logits [B, num_classes]
```

Here, `B`, `C`, and `L` denote batch size, input channels, and sequence length.
Within each scale, patches are rearranged from
`[B, emb_size, patch_num, patch_len]` to
`[B * emb_size, patch_len, patch_num]`. A linear layer projects the last dimension
to `d_model_patch`, so the Transformer operates on
`[B * emb_size, patch_len, d_model_patch]`.

ConvFFN uses a pointwise convolution, a depthwise temporal convolution, and a
second pointwise convolution, with GELU activations and dropout between projections.
The graph relation multiplier is `relation_decay ** distance`; the default
`relation_decay=1.0` gives an all-ones multiplier.

## Installation

Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Dependencies are PyTorch (`torch>=1.13`), NumPy, pandas, SciPy, scikit-learn, sktime,
tqdm, and TensorBoard. Install a PyTorch build appropriate for your CPU or CUDA
environment. The current dependency file specifies a minimum PyTorch version;
it does not pin a complete experiment environment.

## Data preparation

Download the UEA multivariate datasets in `.ts` format from the
[Time Series Classification archive](https://www.timeseriesclassification.com/).
Place the original training and test files under:

```text
Dataset/
└── UEA/
    └── Multivariate_ts/
        ├── Libras/
        │   ├── Libras_TRAIN.ts
        │   └── Libras_TEST.ts
        └── <DatasetName>/
            ├── <DatasetName>_TRAIN.ts
            └── <DatasetName>_TEST.ts
```

The loader also accepts dataset folders directly under `Dataset/UEA/` when the
`Multivariate_ts` subdirectory is absent. Folder and file names must match the
dataset name, including capitalization.

The repository also contains a segmentation data pipeline; see its
[data preparation instructions](Dataset/Segmentation/Segmentation.Txt).
The commands and preprocessing protocol below describe the UEA pipeline.

## Running experiments

### Single dataset

Run Libras with the current defaults:

```bash
python main.py
```

Select another dataset:

```bash
python main.py --single_dataset Epilepsy
```

The main model and training defaults can also be supplied explicitly:

```bash
python main.py \
  --single_dataset Libras \
  --patch_lens 8,16 \
  --strides 4,8 \
  --transformer_layers 1 \
  --num_heads 8 \
  --dim_ff 128 \
  --conv_ffn_kernel_size 7 \
  --relation_decay 1.0 \
  --dropout 0.3 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --val_interval 1 \
  --key_metric loss \
  --seed 1234
```

### All available datasets

```bash
python main.py --run_all
```

This runs every dataset directory under the resolved UEA data path in alphabetical
order. Each directory must contain its matching `_TRAIN.ts` and `_TEST.ts` files.
`--runs_number True` is an equivalent option; it controls dataset selection,
not the number of repeated experiments.

### CPU and CUDA

```bash
python main.py --gpu -1
```

The default uses the current CUDA device when CUDA is available and otherwise
uses the CPU. To select a particular GPU on a shell supporting environment
variable prefixes, restrict visible devices before launching:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --gpu 0
```

In the current implementation, nonnegative `--gpu` values enable CUDA but do not
set the CUDA device index themselves.

### Training curves

```bash
tensorboard --logdir Results
```

Show all public command-line options with:

```bash
python main.py --help
```

## Default configuration

### Model

| Argument | Default | Description |
| --- | --- | --- |
| `--emb_size` | `16` | Embedding channels after channel aggregation |
| `--d_model_patch` | `16` | Transformer feature dimension |
| `--patch_lens` | `8,16` | Comma-separated patch lengths |
| `--strides` | `4,8` | Corresponding patch strides |
| `--transformer_layers` | `1` | Transformer layers per scale |
| `--num_heads` | `8` | Attention heads |
| `--dim_ff` | `128` | ConvFFN hidden dimension |
| `--conv_ffn_kernel_size` | `7` | Depthwise temporal kernel size |
| `--relation_decay` | `1.0` | Graph temporal relation multiplier base |
| `--padding_patch` | `end` | Replicate-pad the end before patch extraction; alternative: `None` |
| `--dropout` | `0.3` | Transformer dropout probability |
| `--pap_dropout` | `0.0` | Dropout after PAP pooling |

Use equally many patch lengths and strides, with positive integer values.
`d_model_patch` must be divisible by `num_heads`. Use a positive odd
`conv_ffn_kernel_size` to preserve sequence length for the residual connection.

### Training

| Argument or setting | Default | Description |
| --- | --- | --- |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `16` | Samples per batch |
| `--lr` | `0.001` | Learning rate |
| Optimizer | `RAdam` | Repository implementation, selected in `main.py` |
| `--optimizer_weight_decay` | `0.0001` | Optimizer weight decay |
| Loss | Cross-entropy | Computed per sample, then averaged for optimization |
| Gradient clipping | `4.0` | Maximum gradient norm |
| `--val_interval` | `1` | Validation frequency in epochs |
| `--key_metric` | `loss` | Validation checkpoint metric: `loss`, `accuracy`, or `precision` |
| `--seed` | `1234` | Random seed |
| `--gpu` | `0` | Current CUDA device when available; `-1` forces CPU |
| `--torch_threads` | `1` | PyTorch CPU thread count |

The optional `precision` selection metric uses the custom aggregation in
`Models/analysis.py`: classes with no predicted samples are excluded and the
remaining class precisions are weighted by their true support.

### Data and output

| Argument | Default | Description |
| --- | --- | --- |
| `--data_path` | `Dataset/UEA/` | Supported choices: `Dataset/UEA/`, `Dataset/Segmentation/` |
| `--single_dataset` | `Libras` | Dataset for a single-dataset run |
| `--Norm` | `False` | Apply normalization fitted on the training split |
| `--val_ratio` | `0.2` | Fraction of the original training set used for validation |
| `--runs_number` | `False` | Run all datasets when `True` |
| `--run_all` | `False` | Flag alias for `--runs_number True` |
| `--output_dir` | `Results` | Root output directory |

## UEA preprocessing and evaluation protocol

1. Read the original `TRAIN` and `TEST` files. Fit the label encoder on the
   original training labels and apply it to test labels.
2. Derive the maximum sequence length from the original training file. Truncate
   longer sequences and pad shorter sequences to this length. Missing or padded
   values are filled with random values in `[0, 0.001)`; an entirely missing
   channel within a sample is filled with zeros.
3. Hold out 20% of the original training samples for validation using the configured
   seed. The loader attempts a stratified split and falls back to a seeded random
   split if stratification raises a `ValueError`.
4. Leave normalization disabled by default. With `--Norm True`, subtract the
   per-channel training mean and divide by the maximum per-sample temporal
   standard deviation for that channel. Zero scaling factors are replaced with
   one. The same training-derived statistics are used for validation and test.
5. Train for the configured number of epochs. Validate every epoch by default;
   the first and final epochs are always validated.
6. Save the checkpoint with the lowest validation loss by default. With
   `--key_metric accuracy` or `precision`, select the highest validation value.
7. Reload the selected checkpoint and evaluate it on the original test split.

Random seeds are initialized once per process for Python, NumPy, and PyTorch.
For separate repeated experiments, launch a new process with each chosen seed.
In an all-dataset run, later datasets share the advancing random state, so their
results can differ from separate single-dataset runs with the same seed.

## Outputs

Each dataset run creates a timestamped directory:

```text
Results/<DatasetName>/<YYYY-MM-DD_HH-MM-SS>/
├── configuration.json
├── checkpoints/
│   └── <DatasetName>_best.pth
├── predictions/                 # Reserved directory
├── tb_summaries/
└── MSPLGFormer_Results_<DatasetName>_<YYYY-MM-DD_HH-MM-SS>.csv
```

`configuration.json` records the effective arguments, dataset dimensions and split
sizes, model summary, parameter counts, optimizer and loss names, environment
information, and output paths. Checkpoints contain the model state, epoch, and
optimizer state. TensorBoard logs contain training loss and validation metrics.

The dataset CSV contains `dataset`, `test_accuracy`, `best_val_epoch`, and
`best_val_loss`. Accuracy is stored as a fraction in `[0, 1]`. Test metrics and the
confusion matrix are also printed to the console.

An all-dataset run additionally writes a combined CSV:

```text
Results/all_datasets/<time>/MSPLGFormer_Results_all_<time>.csv
```

For paper experiments, retain the corresponding `configuration.json` files,
checkpoints, result CSVs, and exact software environment for each reported run.

## Repository structure

```text
main.py                       Command-line entry point and experiment orchestration
Training.py                   Training, validation, and checkpoint selection
utils.py                      Configuration, initialization, and dataset utilities
Models/model.py               MSPLGFormer and its component modules
Models/optimizers.py          Optimizer implementations
Models/loss.py                Classification loss
Models/analysis.py            Classification metrics
Models/utils.py               Checkpoint utilities
Dataset/load_UEA_data.py      UEA loading and preprocessing
Dataset/load_segment_data.py Segmentation loading and preprocessing
requirements.txt              Python dependencies
LICENSE                       License and upstream copyright notice
```

## Acknowledgments and license

The data utilities retain author credits to Chang Wei Tan and Navid Foumani;
`Dataset/classifier_tools.py` also credits
[hfawaz/dl-4-tsc](https://github.com/hfawaz/dl-4-tsc).
The RAdam implementation identifies
[LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam) as its source.
The project includes an MIT license with the upstream Monash University copyright
notice; see [LICENSE](LICENSE).
