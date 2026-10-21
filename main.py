import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import (
    Data_Loader,
    Data_Verifier,
    Initialization,
    Setup,
    collect_environment_info,
    dataset_class,
    prepare_run_dirs,
    save_configuration,
    str2bool,
)
from Models.loss import get_loss_module
from Models.model import count_parameters, model_factory
from Models.optimizers import get_optimizer
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner

warnings.filterwarnings('ignore')
logger = logging.getLogger('__main__')


def build_parser():
    parser = argparse.ArgumentParser(description='MSPLGFormer')
    parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'})
    parser.add_argument('--output_dir', default='Results')
    parser.add_argument('--Norm', type=str2bool, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--runs_number', type=str2bool, default=False, help='True: run all datasets; False: run one dataset')
    parser.add_argument('--run_all', action='store_true', help='Alias for --runs_number True')

    # Embedding, graph, and pooling settings.
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--d_model_patch', type=int, default=16)
    parser.add_argument('--relation_decay', type=float, default=1.0, help='Temporal decay used by the MPNN relation mask')
    parser.add_argument('--padding_patch', choices={'end', 'None'}, default='end')
    parser.add_argument('--pap_dropout', type=float, default=0.0)

    # Multi-scale patch settings.
    parser.add_argument('--patch_lens', type=str, default='8,16', help='Comma-separated patch lengths')
    parser.add_argument('--strides', type=str, default='4,8', help='Comma-separated strides for each patch length')
    parser.add_argument('--patch_len', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--stride', type=int, default=None, help=argparse.SUPPRESS)

    # Lightweight Transformer settings.
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of lightweight Transformer encoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads; d_model_patch must be divisible by num_heads')
    parser.add_argument('--dim_ff', type=int, default=128, help='Hidden dimension of ConvFFN')
    parser.add_argument('--conv_ffn_kernel_size', type=int, default=7, help='Depthwise temporal kernel size in ConvFFN')

    # Training settings.
    parser.add_argument('--single_dataset', type=str, default='Libras')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer_weight_decay', type=float, default=1e-4, help='Weight decay used by the optimizer')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on validation every N epochs. Must be >= 1')
    parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss')
    parser.add_argument('--gpu', type=int, default=0, help='-1: CPU; otherwise use the current CUDA device when available. Select a GPU with CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--torch_threads', default=1, type=int, help='Number of CPU threads used by PyTorch; nonpositive values keep the runtime setting')
    return parser


def make_loaders(data, batch_size):
    train_dataset = dataset_class(data['train_data'], data['train_label'])
    val_dataset = dataset_class(data['val_data'], data['val_label'])
    test_dataset = dataset_class(data['test_data'], data['test_label'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def run_dataset(base_config, problem, device):
    config = base_config.copy()
    config['dataset'] = problem
    config['data_dir'] = os.path.join(config['data_path'], problem)
    prepare_run_dirs(config, problem)

    print(f'Now run the dataset: {problem}')
    logger.info('Loading data ...')
    data = Data_Loader(config)
    train_loader, val_loader, test_loader = make_loaders(data, config['batch_size'])

    config['Data_shape'] = list(data['train_data'].shape)
    config['train_size'] = int(len(data['train_label']))
    config['val_size'] = int(len(data['val_label']))
    config['test_size'] = int(len(data['test_label']))
    config['num_channels'] = int(data['train_data'].shape[1])
    config['sequence_length'] = int(data['train_data'].shape[2])
    config['num_labels'] = int(np.max(data['train_label'])) + 1
    logger.info('Sequence length: %s', config['sequence_length'])
    logger.info('Creating model ...')
    model = model_factory(config).to(device)
    logger.info('Model:\n%s', model)

    config['model_name'] = model.__class__.__name__
    config['model_summary'] = str(model)
    config['model_parameters'] = int(count_parameters(model))
    config['total_parameters'] = int(sum(p.numel() for p in model.parameters()))
    config['trainable_parameters'] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Total number of parameters: %s', config['trainable_parameters'])

    optim_class = get_optimizer('RAdam')
    config['optimizer_name'] = optim_class.__name__
    config['loss_module'] = get_loss_module()
    config['loss_module_name'] = config['loss_module'].__class__.__name__
    config['optimizer'] = optim_class(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['optimizer_weight_decay'],
    )
    save_path = os.path.join(config['save_dir'], f'{problem}_best.pth')
    config['best_checkpoint_path'] = save_path
    config['environment'] = collect_environment_info(device)
    save_configuration(config)

    trainer = SupervisedTrainer(
        model, train_loader, device, config['loss_module'], config['optimizer'],
        l2_reg=0, print_conf_mat=False,
    )
    val_evaluator = SupervisedTrainer(
        model, val_loader, device, config['loss_module'],
        print_conf_mat=False,
    )

    logger.info('Starting training ...')
    best_val_metrics = train_runner(config, model, trainer, val_evaluator, save_path)
    best_model, _, _ = load_model(model, save_path, config['optimizer'])
    best_model.to(device)

    test_evaluator = SupervisedTrainer(
        best_model, test_loader, device, config['loss_module'],
        print_conf_mat=True,
    )
    test_metrics, all_metrics = test_evaluator.evaluate(keep_all=True)
    print_str = 'Best Model Test Summary: '
    for k, v in test_metrics.items():
        print_str += f'{k}: {v} | '
    print(print_str)

    row = {
        'dataset': problem,
        'test_accuracy': all_metrics['total_accuracy'],
        'best_val_epoch': best_val_metrics.get('epoch'),
        'best_val_loss': best_val_metrics.get('loss'),
    }
    # Keep the dataset-level result CSV together with this dataset's run directory:
    # Results/<DatasetName>/<time>/MSPLGFormer_Results_<DatasetName>_<time>.csv
    dataset_results_path = os.path.join(
        config['output_dir'],
        f'MSPLGFormer_Results_{problem}_{config["run_timestamp"]}.csv',
    )
    pd.DataFrame([row]).to_csv(dataset_results_path, index=False)
    config['results_csv_path'] = dataset_results_path
    save_configuration(config)
    print(f'Dataset result saved: {dataset_results_path}')
    return row, dataset_results_path


def main():
    args = build_parser().parse_args()
    config = Setup(args)
    device = Initialization(config)
    Data_Verifier(config)

    if config['runs_number']:
        problems = sorted(
            name for name in os.listdir(config['data_path'])
            if os.path.isdir(os.path.join(config['data_path'], name))
        )
    else:
        problems = [config['single_dataset']]

    all_rows = []
    dataset_result_paths = []
    for problem in problems:
        row, dataset_results_path = run_dataset(config, problem, device)
        all_rows.append(row)
        dataset_result_paths.append(dataset_results_path)

    if config['runs_number']:
        summary_dir = os.path.join(config['root_output_dir'], 'all_datasets', config['run_timestamp'])
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(
            summary_dir,
            f'MSPLGFormer_Results_all_{config["run_timestamp"]}.csv',
        )
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        print(f'All-dataset summary saved: {summary_path}')

    print('Dataset CSV files:')
    for path in dataset_result_paths:
        print(f'  {path}')


if __name__ == '__main__':
    main()
