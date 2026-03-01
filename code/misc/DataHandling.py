import glob
import json
import os
import re
from datetime import datetime
import misc.ImageUtils as iu
import numpy as np

def _sanitize_token(value):
    token = re.sub('[^0-9a-zA-Z]+', '_', str(value).strip()).strip('_')
    return token.lower() if token else 'unknown'

def _network_module_tag(network_module):
    return _sanitize_token(str(network_module).split('.')[-1])

def _resolve_experiment_name(args):
    explicit_name = getattr(args, 'ExperimentFileName', 'default')
    if explicit_name and explicit_name != 'default':
        return explicit_name
    dataset_tag = _sanitize_token(getattr(args, 'Dataset', 'dataset'))
    module_tag = _network_module_tag(getattr(args, 'network_module', 'network'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{module_tag}_{dataset_tag}_{timestamp}'

def _build_run_manifest(args):
    return {'network_module': getattr(args, 'network_module', 'network.MultiScaleResNet'), 'dataset': args.Dataset, 'num_out': int(args.NumOut), 'init_neurons': int(getattr(args, 'EffectiveInitNeurons', 32)), 'num_subblocks': int(getattr(args, 'EffectiveNumSubBlocks', 2)), 'expansion_factor': float(getattr(args, 'EffectiveExpansionFactor', 2.0)), 'uncertainty_type': getattr(args, 'EffectiveUncertaintyType', 'LinearSoftplus')}

def WriteRunManifest(args):
    manifest_path = os.path.join(args.CheckPointPath, 'run_manifest.json')
    payload = _build_run_manifest(args)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write('\n')
    return manifest_path

def ValidateResumeManifest(args):
    manifest_path = os.path.join(args.ResumeCheckPointPath, 'run_manifest.json')
    if not os.path.isfile(manifest_path):
        print(f'ERROR: resume manifest not found: {manifest_path}')
        raise SystemExit(1)
    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            resume_manifest = json.load(manifest_file)
    except Exception as exc:
        print(f"ERROR: failed to read resume manifest '{manifest_path}': {exc}")
        raise SystemExit(1) from exc
    current_manifest = _build_run_manifest(args)
    mismatches = []
    for key in sorted(current_manifest.keys()):
        resume_value = resume_manifest.get(key, None)
        current_value = current_manifest[key]
        if resume_value != current_value:
            mismatches.append((key, resume_value, current_value))
    if mismatches:
        print('ERROR: resume manifest mismatch detected:')
        for key, resume_value, current_value in mismatches:
            print(f"  - {key}: resume='{resume_value}' current='{current_value}'")
        raise SystemExit(1)

def FindLatestModel(checkpoint_path):
    file_list = glob.glob(os.path.join(checkpoint_path, '*.ckpt.index'))
    if not file_list:
        raise FileNotFoundError(f'No checkpoint files found in {checkpoint_path}')
    latest_file = max(file_list, key=os.path.getctime)
    latest_file = os.path.basename(latest_file)
    latest_file = latest_file.replace('.ckpt.index', '')
    return latest_file

def SetupAll(args):
    dir_names_path = f'{args.data_list}/{args.Dataset}_dirnames.txt'
    train_path = f'{args.data_list}/{args.Dataset}_train.txt'
    val_path = f'{args.data_list}/{args.Dataset}_val.txt'
    test_path = f'{args.data_list}/{args.Dataset}_test.txt'
    dir_names, train_names, val_names, test_names = ReadDirNames(dir_names_path, train_path, val_path, test_path)
    use_default_flag = 0
    if use_default_flag:
        optimizer_params = [0.001, 0.9, 0.999, 1e-08]
    else:
        optimizer_params = [args.LR, 0.9, 0.999, 1e-08]
    save_checkpoint = 1000
    num_test_runs_per_epoch = 5
    if args.Dataset == 'FT3D':
        original_image_size = np.array([540, 960, 3])
        patch_size = np.array([480, 640, 3])
        num_out = 2
    elif args.Dataset == 'FC2':
        original_image_size = np.array([384, 512, 3])
        patch_size = np.array([352, 480, 3])
        num_out = 2
    elif args.Dataset == 'MSCOCO':
        original_image_size = np.array([504, 640, 3])
        patch_size = np.array([352, 480, 3])
        num_out = 3
    else:
        raise ValueError(f'Unsupported dataset: {args.Dataset}')
    num_train_samples = len(train_names)
    num_val_samples = len(val_names)
    num_test_samples = len(test_names)
    lam = np.array([1.0, 1.0])
    checkpoint_root = os.path.abspath('./Checkpoints')
    logs_root = os.path.abspath('./Logs')
    experiment_name = _resolve_experiment_name(args)
    args.ExperimentName = experiment_name
    args.CheckPointRoot = checkpoint_root
    args.LogsRoot = logs_root
    args.CheckPointPath = os.path.join(checkpoint_root, experiment_name)
    args.LogsPath = os.path.join(logs_root, experiment_name)
    args.TrainNames = train_names
    args.ValNames = val_names
    args.TestNames = test_names
    args.OptimizerParams = optimizer_params
    args.SaveCheckPoint = save_checkpoint
    args.PatchSize = patch_size
    args.NumTrainSamples = num_train_samples
    args.NumValSamples = num_val_samples
    args.NumTestSamples = num_test_samples
    args.NumTestRunsPerEpoch = num_test_runs_per_epoch
    args.OriginalImageSize = original_image_size
    args.Lambda = lam
    args.NumOut = num_out
    args.LossFuncName = 'MultiscaleSL1-1'
    if not os.path.isdir(args.CheckPointPath):
        os.makedirs(args.CheckPointPath)
    if not os.path.isdir(args.LogsPath):
        os.makedirs(args.LogsPath)
    if args.LoadCheckPoint == 1:
        resume_experiment_name = getattr(args, 'ResumeExperimentFileName', '')
        if not resume_experiment_name:
            print('ERROR: --LoadCheckPoint=1 requires --ResumeExperimentFileName.')
            raise SystemExit(1)
        args.ResumeExperimentName = resume_experiment_name
        args.ResumeCheckPointPath = os.path.join(checkpoint_root, resume_experiment_name)
        if not os.path.isdir(args.ResumeCheckPointPath):
            print(f'ERROR: resume checkpoint directory not found: {args.ResumeCheckPointPath}')
            raise SystemExit(1)
        try:
            args.LatestFile = FindLatestModel(args.ResumeCheckPointPath)
        except FileNotFoundError as exc:
            print(f'ERROR: {exc}')
            raise SystemExit(1) from exc
    else:
        args.ResumeExperimentName = None
        args.ResumeCheckPointPath = None
        args.LatestFile = None
    return args

def ReadDirNames(dir_names_path, train_path, val_path, test_path):
    dir_names = open(dir_names_path, 'r')
    dir_names = dir_names.read()
    dir_names = dir_names.split()
    train_idxs = open(train_path, 'r')
    train_idxs = train_idxs.read()
    train_idxs = train_idxs.split()
    train_idxs = [int(val) for val in train_idxs]
    train_names = [dir_names[i] for i in train_idxs]
    val_idxs = open(val_path, 'r')
    val_idxs = val_idxs.read()
    val_idxs = val_idxs.split()
    val_idxs = [int(val) for val in val_idxs]
    val_names = [dir_names[i] for i in val_idxs]
    test_idxs = open(test_path, 'r')
    test_idxs = test_idxs.read()
    test_idxs = test_idxs.split()
    test_idxs = [int(val) for val in test_idxs]
    test_names = [dir_names[i] for i in test_idxs]
    return (dir_names, train_names, val_names, test_names)