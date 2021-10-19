from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from utils.Compare_Func import cifar10_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
import torch

torch.backends.cudnn.benchmark = True

seed_everything(17211401)

dev = torch.device("cuda")
dataset_train = cifar_dataset("datasets", True)
dataloader_train = DataLoader(dataset_train, batch_size=2048, shuffle=True, num_workers=48)
dataset_test = cifar_dataset("datasets", True, train=False)
dataloader_test = DataLoader(dataset_test, batch_size=2048, shuffle=True, num_workers=48)

def get_mine_net():
    from model.SwinTransformer import SwinTransformer
    net = SwinTransformer(
        image_size = 32,
        image_channel = 3,
        patch_size = 2,
        embed_dim = 18, 
        window_size = 8,
        classes_num = 10, 
        heads_num_list = [3, 6, 12, 24],
        depth_num_list = [2, 2, 6, 2],
        wsa_dropout = 0.2,
        mlp_dropout = 0.2,
        droppath = 0.2,
        qkv_bias = True,
        mlp_inner_ratio = 4
    )
    net = torch.nn.DataParallel(net).to(dev)


    logger = Logger("save/Mine-SwinTransformer_test_batch-size-2048", net, load_newest=True)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    trainer = Trainer(net, loss, opt, dev, logger, 10)

    return (net, logger, trainer)

def get_ref_net():
    from ref_model.SwinTransformer import SwinTransformer
    net = SwinTransformer(
        img_size = 32,
        in_chans = 3,
        patch_size = 2,
        embed_dim = 18,
        window_size = 8,
        num_classes = 10,
        num_heads = [3, 6, 12, 24],
        depths = [2, 2, 6, 2],
        attn_drop_rate = 0.2,
        drop_path_rate = 0.2,
        qkv_bias = True,
        mlp_ratio = 4
    )
    net = torch.nn.DataParallel(net).to(dev)


    logger = Logger("save/Ref-SwinTransformer_embed_dim-18", net, load_newest=True)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    trainer = Trainer(net, loss, opt, dev, logger, 10)
    return (net, logger, trainer)
