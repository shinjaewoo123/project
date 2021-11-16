import os
import argparse
import yaml
import easydict

def read_cfg(path='', isjupyter=False):
    if isjupyter:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        args = easydict.EasyDict(**cfg)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/cfg_efficientnetv2.yaml', 
                help='for read args from config.yaml file')

        args, remaining = parser.parse_known_args()
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args(remaining)

    return args


def write_cfg(args):
    ## 로그 경로에 cfg 파일 저장.
    cfg_filename = os.path.basename(args.config)
    log_cfg_path = os.path.join(args.model_save_dir, cfg_filename) 
    
    with open(log_cfg_path, 'w') as f:
        yaml.dump(vars(args), f)

