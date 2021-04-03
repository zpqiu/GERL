# -*- encoding:utf-8 -*-
"""
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
import argparse
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from datasets.dataset import TrainingDataset
from datasets.dataset import ValidationDataset
from models.gerl import Model
# from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric


def run(cfg: DictConfig, rank: int, device: torch.device, corpus_path: str):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    set_seed(cfg.training.seed)

    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    valid_dataset = ValidationDataset(cfg.dataset, corpus_path)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=cfg.training.batch_size)

    # Build model.
    model = Model(cfg)

    saved_model_path = os.path.join(cfg.training.model_save_path, 'model.ep{0}'.format(cfg.training.validate_epoch))
    print("Load from:", saved_model_path)
    if not os.path.exists(saved_model_path):
        print("Not Exist: {}".format(saved_model_path))
        return
    model.cpu()
    pretrained_model = torch.load(saved_model_path, map_location='cpu')
    model.load_state_dict(pretrained_model, strict=False)
    model.title_encoder.title_embedding = model.title_encoder.title_embedding.to(device)
    model.to(device)
    model.eval()
    
    print("Worker %d is working ... " % rank)
    validate(cfg, rank, model, valid_data_loader, device)


def validate(cfg, rank, model, valid_data_loader, device):
    model.eval()

    # Setting the tqdm progress bar
    if rank == 0:
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test:",
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)

    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:
            imp_ids += data["imp_id"].cpu().numpy().tolist()
            data ={key: value.to(device) for key, value in data.items()}

            # 1. Forward
            pred = model.prediction_step(data)

            preds += pred.cpu().numpy().tolist()
            truths += data["y"].long().cpu().numpy().tolist()

        result_file_path = os.path.join(cfg.dataset.result_path, "split_{}.txt".format(rank))

        with open(result_file_path, 'w', encoding='utf-8') as f:
            for imp_id, truth, pred in zip(imp_ids, truths, preds):
                f.write("{}\t{}\t{}\n".format(imp_id, truth, pred))


def init_processes(cfg, local_rank, dataset, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = cfg.training.master_port
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.training.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, corpus_path=dataset)


def set_log_service(api_token, params, project_name="alexchiu/mind", exp_name="base_gat"):
    train_dir = os.path.dirname(__file__)
    # neptune.init(project_name, api_token=api_token)
    # neptune.create_experiment(name=exp_name, params=params, upload_source_files=[train_dir + '/models/kg_model.py'])


@hydra.main(config_path="../conf/train.yaml")
def main(cfg):
    # init_exp(cfg)
    set_seed(cfg.training.seed)

    if cfg.training.gpus == 0:
        print("== CPU Mode ==")
        datasets = cfg.dataset.train
        run(cfg, 0, torch.device("cpu"), datasets)
    elif cfg.training.gpus == 1:
        datasets = cfg.dataset.train
        run(cfg, 0, torch.device("cuda:0"), datasets)
    else:
        processes = []
        for rank in range(cfg.training.gpus):
            datasets = cfg.dataset.test + ".p{}.tsv".format(rank)
            p = mp.Process(target=init_processes, args=(
                cfg, rank, datasets, run, "nccl"))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
