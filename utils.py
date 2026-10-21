import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from importlib import metadata

import numpy as np
import torch
from torch.utils.data import Dataset


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {'yes', 'true', 't', '1', 'y'}:
        return True
    if v in {'no', 'false', 'f', '0', 'n'}:
        return False
    raise ValueError(f'Invalid boolean value: {v}')


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    value = str(value).strip()
    if not value:
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def Setup(args):
    config = vars(args).copy()
    if config.get('run_all'):
        config['runs_number'] = True

    config['patch_lens'] = parse_int_list(config['patch_lens'])
    config['strides'] = parse_int_list(config['strides'])

    if not config['patch_lens'] and config.get('patch_len') is not None:
        config['patch_lens'] = [int(config['patch_len'])]
    if not config['strides'] and config.get('stride') is not None:
        config['strides'] = [int(config['stride'])]
    if len(config['patch_lens']) != len(config['strides']):
        raise ValueError('--patch_lens and --strides must contain the same number of comma-separated values')
    if config['val_interval'] < 1:
        raise ValueError('--val_interval must be >= 1')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config['run_timestamp'] = timestamp
    # Group outputs under <output_dir>/<DatasetName>/<time>/.
    config['root_output_dir'] = config['output_dir']
    os.makedirs(config['root_output_dir'], exist_ok=True)
    return config


def prepare_run_dirs(config, problem):
    output_dir = os.path.join(config['root_output_dir'], problem, config['run_timestamp'])
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])


def json_safe(value):
    """Convert common Python/Numpy/PyTorch values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info(device=None):
    cuda_available = torch.cuda.is_available()
    env = {
        'python_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'pandas_version': package_version('pandas'),
        'scikit_learn_version': package_version('scikit-learn'),
        'sktime_version': package_version('sktime'),
        'cuda_available': cuda_available,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else None,
        'mps_available': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
        'torch_num_threads': torch.get_num_threads(),
        'device': str(device) if device is not None else None,
    }
    if cuda_available:
        env['cuda_device_count'] = torch.cuda.device_count()
        env['cuda_device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env['current_cuda_device'] = torch.cuda.current_device()
    else:
        env['cuda_device_count'] = 0
        env['cuda_device_names'] = []
        env['current_cuda_device'] = None
    return env


def save_configuration(config, output_dir=None):
    """Save run settings, metadata, and collected environment information as JSON.

    Runtime objects such as the optimizer and loss module are intentionally
    excluded; their names are stored separately in optimizer_name and
    loss_module_name.
    """
    output_dir = output_dir or config['output_dir']
    excluded_keys = {'optimizer', 'loss_module'}
    config_to_save = {k: v for k, v in config.items() if k not in excluded_keys}
    path = os.path.join(output_dir, 'configuration.json')
    with open(path, 'w') as fp:
        json.dump(json_safe(config_to_save), fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '%s'", path)
    return path


def create_dirs(dirs):
    for dir_ in dirs:
        os.makedirs(dir_, exist_ok=True)


def Initialization(config):
    if config.get('torch_threads') is not None and int(config['torch_threads']) > 0:
        torch.set_num_threads(int(config['torch_threads']))
    seed = config.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available() and str(config['gpu']) != '-1'
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info('Using device: %s', device)
    if use_cuda:
        logger.info('Device index: %s', torch.cuda.current_device())
    return device


def Data_Loader(config):
    if 'Segmentation' in config['data_path']:
        from Dataset import load_segment_data
        return load_segment_data.load(config)
    from Dataset import load_UEA_data
    return load_UEA_data.load(config)


def Data_Verifier(config):
    if not os.path.exists(config['data_path']):
        raise FileNotFoundError(
            f"Data path not found: {config['data_path']}. Put the UEA data under Dataset/UEA/Multivariate_ts/"
        )
    if 'UEA' in config['data_path']:
        multivariate_path = os.path.join(config['data_path'], 'Multivariate_ts')
        if os.path.isdir(multivariate_path):
            config['data_path'] = multivariate_path
        logger.info('Using UEA data path: %s', config['data_path'])


class dataset_class(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.feature = data
        self.labels = label.astype(np.int64)

    def __getitem__(self, ind):
        x = torch.tensor(self.feature[ind].astype(np.float32))
        y = torch.tensor(self.labels[ind], dtype=torch.long)
        return x, y, ind

    def __len__(self):
        return len(self.labels)
