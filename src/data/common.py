from importlib import import_module


def get_dataloader(args):
    return import_module('data.' + args.dataset.lower()).DataLoader(args)

class AbstractDataLoader():
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.c = args.n_colors
        self.class_num = args.class_num
        self.seed = args.seed
        args.task_dir = args.data_dir + '/segmentation/'
        self.type = 'jpg'
        self.img_dir = args.task_dir + '/images/'
        self.gt_dir = args.task_dir + '/gts/'
        self.ori_gt_type = 'mat'
        self.gt_type = 'mat'
        self.hdf5_dir = args.task_dir + '/hdf5'

    def prepare_dataset(self):
        # Use your own prepare function here
        pass