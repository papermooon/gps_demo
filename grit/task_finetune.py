import argparse

import torch
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
# from .dataset.dataset import create_dataset
from grit.dataset.mini_dataset import create_miniDataset
from grit.model.grit_model import GritTransformer
# from .model.cl_model import CLModel
from task_trainer import TaskTrainer


def get_dataloaders(config):
    # train_set, val_set, test_set = create_dataset(config)
    train_set, val_set, test_set = create_miniDataset(config)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data)
    # nout = 10 # len(config.data.target_col.split(','))
    nout = 2 # len(config.data.target_col.split(','))
    model = GritTransformer(nout, **config.model.grit, ksteps=config.data.pos_enc_rrwp.ksteps)
    from torchinfo import summary
    summary(model)

    if args.ckpt is not None:
       model = load_model(model, args.ckpt)

    # if args.ckpt_cl is not None:
    #    cl_model = CLModel(model, **config.model.cl_model)
    #    cl_model = load_model(cl_model, args.ckpt_cl)
    #    model = cl_model.encoder

    # exit()
    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, val_dataloader, test_dataloader)
    logger.info(f"Training finished")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='grit/task_finetune.yaml')
    parser.add_argument('--config', default='task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/grit/task_finetune')
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--ckpt_cl', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = parse_config(args.config)
    main(args, config)
