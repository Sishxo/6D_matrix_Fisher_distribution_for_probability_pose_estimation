import matplotlib.pyplot as plt
import numpy as np
import tqdm
from Pascal3D import Pascal3D, Pascal3D_render, Pascal3D_all
from ModelNetSo3 import ModelNetSo3
from network.resnet import resnet50, resnet101, ResnetHead
from network.Fisher_n6d import Fisher_n6d
from network.rot_head import RotHeadNet
from UPNA import UPNA
from loss import vmf_loss,sampling_loss_Rest
import torch
import torch.nn as nn
import os
import tqdm
import argparse
import utils.dataloader_utils as dataloader_utils
import logger
import matplotlib
import json
from datetime import datetime
import pytz
from utils.rot_utils import get_rot_vec_vert_batch
import time
from tqdm import tqdm
import timm

matplotlib.use("Agg")

dataset_dir = "datasets"  # TODO change with dataset path


def get_pascal_no_warp_loaders(batch_size, train_all, voc_train, source, category=None):
    dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=False, voc_train=voc_train, source=source, category=category)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def get_pascal_loaders(batch_size, train_all, use_synthetic_data, use_augment, voc_train, source, category=None):
    if use_synthetic_data:
        return get_pascal_synthetic(batch_size, train_all, use_augment, voc_train, source, category)
    else:
        dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train, source=source, category=category)
        dataloader_train = torch.utils.data.DataLoader(
            dataset.get_train(use_augment),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
            pin_memory=True,
            drop_last=True)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset.get_eval(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
            pin_memory=True,
            drop_last=False)
        return dataloader_train, dataloader_eval


def get_pascal_synthetic(batch_size, train_all, use_augmentation, voc_train, source, category):
    dataset_real = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train, source=source, category=category)
    train_real = dataset_real.get_train(use_augmentation)
    real_sampler = torch.utils.data.sampler.RandomSampler(train_real, replacement=False)
    dataset_rendered = Pascal3D_render.Pascal3DRendered(dataset_dir, category=category)
    rendered_size = int(0.2 * len(dataset_rendered))  # use 20% of synthetic data for training per epoch
    rendered_sampler = dataloader_utils.RandomSubsetSampler(dataset_rendered, rendered_size)
    dataset_train, sampler_train = dataloader_utils.get_concatenated_dataset([(train_real, real_sampler), (dataset_rendered, rendered_sampler)])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_real.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

def get_upna_loaders(batch_size, train_all):
    dataset = UPNA.n(dataset_dir)
    train_ds = dataset.get_train()
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)
        ),
        pin_memory=True,
        drop_last=True,
    )

    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)
        ),
        pin_memory=True,
        drop_last=False,
    )
    return dataloader_train, dataloader_eval


def get_modelnet_loaders(batch_size, train_all, category=None):
    dataset = ModelNetSo3.ModelNetSo3(dataset_dir,category)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # I suspect the suprocess fails to free their transactions when terminating? not too much processing done in dataloader anyways
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)
        ),
        pin_memory=True,
        drop_last=True,
    )
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # I suspect the suprocess fails to free their transactions when terminating? not too much processing done in dataloader anyways
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)
        ),
        pin_memory=True,
        drop_last=False,
    )
    return dataloader_train, dataloader_eval


def train_model(train_setting):
    # device = 'cpu'
    device = "cuda"
    batch_size = train_setting.batch
    train_all = True  # train_all=False when decisions were made
    config = train_setting.config
    run_name = train_setting.run_name
    gpus = train_setting.gpus
    category = train_setting.category
    learning_rate = train_setting.lr
    loss = train_setting.loss
    net = train_setting.net
    out_dim = train_setting.out_dim
    optimizer = train_setting.opt
    
    if loss=="sampling":
        loss_func=sampling_loss_Rest
    elif loss=="vmf":
        loss_func=vmf_loss
    else:
        raise ValueError("no such loss named",loss)
    
    if net.lower()=="vit":
        base = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            pretrained_cfg_overlay=dict(file='/data0/sunshichu/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin'),
        )
    elif net.lower()=="resnet":
        base = resnet101(pretrained=True, progress=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus

    if config.type == "pascal":
        num_classes = 12 + 1  # +1 due to one indexed classes
    elif config.type == "modelnet":
        num_classes = 10
    else:
        raise ValueError("no such dataset")

    
    fisher_head = ResnetHead(
            base, num_classes, config.embedding_dim, 512, out_dim
    )
    #rot_head = RotHeadNet(base.output_size)
    #model = Fisher_n6d(base, fisher_head, rot_head, batch_size)
    model = fisher_head
    
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    
    model.to(device)

    
    if config.type == "pascal":
        use_synthetic_data = config.synthetic_data
        use_augmentation = config.data_aug
        use_warp = config.warp
        voc_train = config.pascal_train
        source = config.source
        if not use_warp:
            assert not use_synthetic_data
            assert not use_augmentation
            dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(
                batch_size, train_all, voc_train, source, category
            )
        else:
            dataloader_train, dataloader_eval = get_pascal_loaders(
                batch_size, train_all, use_synthetic_data, use_augmentation, voc_train, source, category
            )
    elif config.type == "modelnet":
        dataloader_train, dataloader_eval = get_modelnet_loaders(batch_size, train_all, category)
    elif config.type == "upna":
        dataloader_train, dataloader_eval = get_upna_loaders(batch_size, train_all)
    else:
        raise Exception("Unknown config: {}".config.format())


    if isinstance(model, nn.DataParallel):
        if model.module.class_embedding:
            finetune_parameters = list(model.module.head.parameters())+list(model.module.class_embedding.parameters()) 
        else:
            finetune_parameters = model.module.head.parameters()
    else:
        if model.class_embedding:
            finetune_parameters = list(model.head.parameters())+list(model.class_embedding.parameters()) 
        else:
            finetune_parameters = model.head.parameters()
        
    if config.type == "modelnet":
        num_epochs = 50
        drop_epochs = [30, 40, 45, np.inf]
        stop_finetune_epoch = 2
    else:
        num_epochs = 120
        drop_epochs = [30, 60, 90, np.inf]
        stop_finetune_epoch = 3
    drop_idx = 0
    
    grids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eq_grids', 'grids3.npy')
    print(f'Loading SO3 discrete grids {grids_path}')
    grids = torch.from_numpy(np.load(grids_path)).to(device)

    cur_lr = learning_rate
    if optimizer.lower()=='sgd':
        opt = torch.optim.SGD(finetune_parameters, lr=cur_lr)
    elif optimizer.lower()=='adam':
        opt = torch.optim.Adam(finetune_parameters,lr=cur_lr)
    if config.type == "pascal":
        class_enum = Pascal3D.PascalClasses
    else:
        class_enum = ModelNetSo3.ModelNetSo3Classes
        
    log_dir = "logs/{}/{}".format(config.type, run_name)
    loggers = logger.Logger(log_dir, class_enum, config=config, train_setting=train_setting)
    
    for epoch in range(num_epochs):
        read_data_start_time = time.time()
        verbose = epoch % 20 == 0 or epoch == num_epochs - 1
        
        if optimizer.lower()=='sgd':
            if epoch == drop_epochs[drop_idx]:
                cur_lr *= 0.1
                drop_idx += 1
                opt = torch.optim.SGD(model.parameters(), lr=cur_lr)
            elif epoch == stop_finetune_epoch:
                opt = torch.optim.SGD(model.parameters(), lr=cur_lr)
        
        logger_train = loggers.get_train_logger(epoch, verbose)

        model.train()
        for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm(dataloader_train):
            image = image.to(device)
            R = extrinsic[:, :3, :3].to(device)
            class_idx = class_idx_cpu.to(device)

            # fisher_output, p_green_R, p_red_R, f_green_R, f_red_R = model(image, class_idx)
            out = model(image,class_idx)
            
            # losses, Rest = loss_func(batch_size,fisher_output, R, f_green_R,f_red_R,p_green_R,p_red_R, overreg=1.05)
            losses, Rest = loss_func(out,R,grids,overreg=1.025)
            
            if losses is not None:
                loss = torch.mean(losses)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
            logger_train.add_samples(
                image, losses, None, R, Rest, class_idx_cpu, hard
            )
        logger_train.finish()
        logger_train = None
        image = None
        R = None
        class_idx = None
        fisher_output = None
        loss_value = None
        Rest = None

        logger_eval = loggers.get_validation_logger(epoch, verbose)
        model.eval()
        with torch.no_grad():
            for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm(dataloader_eval):
                image = image.to(device)
                R = extrinsic[:, :3, :3].to(device)
                class_idx = class_idx_cpu.to(device)
                
                # fisher_output, p_green_R, p_red_R, f_green_R, f_red_R = model(image, class_idx)
                out = model(image,class_idx)
                # losses, Rest = loss_func(batch_size,fisher_output, R, f_green_R,f_red_R,p_green_R,p_red_R, overreg=1.05)
                losses, Rest = loss_func(out, R, grids, overreg=1.025)
                
                if losses is None:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_eval.add_samples(
                    image, losses, None, R, Rest, class_idx_cpu, hard
                )
        logger_eval.finish()
        if verbose:
            loggers.save_network(epoch, model)
        
        read_data_end_time = time.time()
        print("cost time: "+str(read_data_end_time-read_data_start_time))


class TrainSetting:
    def __init__(self, config, args):
        self.run_name = args.run_name
        self.config = config
        self.gpus = args.gpus
        self.batch = args.batch
        self.category = args.category
        self.loss = args.loss
        self.net = args.net
        self.lr = args.lr
        self.out_dim = args.out_dim
        self.opt = args.opt
        print("---- Train Setting -----")
        for k, v in sorted(args.__dict__.items()):
            self.__setattr__(k, v)
            print(f"{k:20}: {v}")

    def json_serialize(self):
        return {
            "run_name": self.run_name,
            "gpus":self.gpus,
            "batch":self.batch,
            "category":self.category,
            "loss":self.loss,
            "net":self.net,
            "lr":self.lr,
            "out_dim":self.out_dim,
            "opt":self.opt
        }

    @staticmethod
    def json_deserialize(dic):
        config = TrainConfig.json_deserialize(dic["config"])
        return TrainSetting(config)


class TrainConfig:
    def __init__(self, typ):
        self.type = typ

    @staticmethod
    def json_deserialize(dic):
        if dic["type"] == "pascal":
            return PascalConfig.json_deserialize(dic)
        elif dic["type"] == "upna":
            return UPNAConfig.json_deserialize(dic)
        elif dic["type"] == "modelnet":
            return ModelnetConfig.json_deserialize(dic)
        else:
            raise RuntimeError("Can not deserialize Train config: {}".format(dic))

    def json_serialize(self):
        raise RuntimeError("can not serialize abstract class")


class PascalConfig(TrainConfig):
    # data_aug is bool
    # embedding_dim is int
    # synthetic_data is bool
    # warp is bool
    def __init__(self, data_aug, embedding_dim, synthetic_data, warp, pascal_train, source):
        super().__init__("pascal")
        self.data_aug = data_aug
        self.embedding_dim = embedding_dim
        self.synthetic_data = synthetic_data
        self.warp = warp
        self.pascal_train = pascal_train
        self.source = source

    @staticmethod
    def json_deserialize(dic):
        data_aug = dic["data_aug"]
        embedding_dim = dic["embedding_dim"]
        synthetic_data = dic["synthetic_data"]
        warp = dic["warp"]
        pascal_train = dic["pascal_train"]
        source = dic["source"]
        return PascalConfig(data_aug, embedding_dim, synthetic_data, warp, pascal_train, source)

    def json_serialize(self):
        return {
            "type": "pascal",
            "data_aug": self.data_aug,
            "embedding_dim": self.embedding_dim,
            "synthetic_data": self.synthetic_data,
            "warp": self.warp,
            "pascal_train": self.pascal_train,
            "source": self.source
        }


class ModelnetConfig(TrainConfig):
    # embedding_dim is int
    def __init__(self, embedding_dim):
        super().__init__("modelnet")
        self.embedding_dim = embedding_dim

    @staticmethod
    def json_deserialize(dic):
        return ModelnetConfig(dic["embedding_dim"])

    def json_serialize(self):
        return {"type": "modelnet", "embedding_dim": self.embedding_dim}


class UPNAConfig(TrainConfig):
    def __init__(self):
        super().__init__("upna")
        self.embedding_dim = 0

    @staticmethod
    def json_deserialize(dic):
        return UPNAConfig()

    def json_serialize(self):
        return {"type": "upna"}


def get_time():
    timezone = pytz.timezone("Asia/Shanghai")
    current_time = datetime.now(timezone)
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")
    return formatted_time


def parse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_name", type=str, default="dummy")
    arg_parser.add_argument("--config_file", type=str)
    arg_parser.add_argument("--gpus",type=str, default='0')
    arg_parser.add_argument("--batch",type=int,default=32)
    arg_parser.add_argument('--category', help='select category for ModelNet and Pascal3D+')
    arg_parser.add_argument('--loss',type=str,default='sampling')
    arg_parser.add_argument('--net',type=str,default='ViT')
    arg_parser.add_argument('--lr', type=float, default=0.01)
    arg_parser.add_argument('--out_dim',type=int,default=6)
    arg_parser.add_argument('--opt',type=str,default='Adam')
    args = arg_parser.parse_args()
    current_time = get_time()
    if args.run_name == "dummy":
        args.run_name = (
            os.path.splitext(os.path.basename(args.config_file))[0] + "_" + current_time
        )
        
    config_file = args.config_file
    with open(config_file, "rb") as f:
        # json_bytes = f.read()
        # json_str = json_bytes.decode('utf-8')
        config_dict = json.load(f)
    config = TrainConfig.json_deserialize(config_dict)
    #gpu_idx = args.gpus.split(',') if args.gpus else []
    #gpu_idx = [int(gpu.strip()) for gpu in gpu_idx]
    training_setting = TrainSetting(config, args)
    return training_setting


import shutil


def main():
    train_setting = parse_config()
    train_model(train_setting)


if __name__ == "__main__":
    main()
