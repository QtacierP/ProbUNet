from option import args
from data.common import get_dataloader
from model.common import get_model
from utils import dataset_normalized, inference_normalized
import os

if __name__ == '__main__':
    data_loader = get_dataloader(args)
    if args.prepare:
        data_loader.prepare()
    train_data, train_label = data_loader.get_train()
    val_data, val_label = data_loader.get_val()
    test_data, test_label = data_loader.get_test()
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_data, mean, std = dataset_normalized(train_data)
    val_data = inference_normalized(val_data, imgs_std=std, imgs_mean=mean)
    test_data = inference_normalized(test_data, imgs_std=std, imgs_mean=mean)
    model = get_model(args)
    model.train(train_data, train_label, val_data, val_label)