import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import argparse
import pathlib
import numpy as np
from utils import train_model, weights_init
from net import CNNforMNIST


if __name__ == '__main__':
    # make parser
    parser = argparse.ArgumentParser(
        prog='classify mnist',
        usage='python train.py',
        description='description',
        epilog='end',
        add_help=True
    )
    # add argument
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=120)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0,'
                        ' -1 is means don\'t use gpu',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)
    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = "result_{0}/result_{0}_{1}".format(number, seed)

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))

    if gpu == 0:
        device = torch.device("cuda:0")
    elif gpu == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # path to data directory
    data_dir = pathlib.Path('data').resolve()
    # transform
    data_transform = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    # load datasets
    mnist_data = {phase: datasets.MNIST(root=data_dir / phase, train=(phase == 'train'),
                                        download=True,
                                        transform=data_transform[phase])
                  for phase in ['train', 'val']}

    # build model
    model = CNNforMNIST().to(device)
    # initilalize parameters of model
    model.apply(weights_init)
    # prepare optimizer & criterion
    optimizer = optim.RMSprop(params=model.parameters(),
                              lr=0.001,
                              alpha=0.9,
                              eps=1e-08)
    criterion = nn.CrossEntropyLoss()
    # train model
    best_model = train_model(model=model, datasets=mnist_data, criterion=criterion,
                             optimizer=optimizer, device=device, num_epochs=epoch, batch_size=batch_size)
