# -*- coding: utf-8 -*-
import argparse
import os

args = argparse.ArgumentParser(description='The option of ProbUNet')

# dataset option
args.add_argument('--dataset', type=str, default='ORIGA', help='Only support ORIGA now')
args.add_argument('--class_num', type=int, default=3, help='')
args.add_argument('--n_colors', type=int, default=3, help='')
args.add_argument('--height', type=int, default=256, help='')
args.add_argument('--width', type=int, default=256, help='')
args.add_argument('--prepare', action='store_true', help='prepare data to hdf5')

# train and test option
args.add_argument('--gpu', type=str, default='5', help='GPU index')
args.add_argument('--seed', type=int, default=0, help='')
args.add_argument('--test', action='store_true', help='test')
args.add_argument('--beta', type=float, default=1.0, help='')
args.add_argument('--batch_size', type=int, default=8, help='')
args.add_argument('--epoch', type=int, default=50, help='')
args.add_argument('--early_stopping', type=int, default=150, help='')
# model option
args.add_argument('--model', type=str, default='ProbUNet', help='')

args = args.parse_args()


args.model_path = '../model/' + args.model + '/'
args.data_dir = '../data/' + args.dataset + '/'

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)
if not os.path.exists(args.model_path + 'log/'):
    os.mkdir(args.model_path + 'log/')