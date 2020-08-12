import os
import argparse
from voc12 import voc_loader
from solver import Solver

def main(args):
    if args.mode == 'train':
        train_loader = voc_loader(args.root, args.split, args.ignore_label, args.mean_bgr, args.augment, 
                                  args.base_size, args.crop_size, args.scales, args.flip, args)
        if args.val:
            val_loader = voc_loader(args.root, 'val', None, args.mean_bgr, False, False, False, False, False, args)
            train = Solver(train_loader, val_loader, None, args)
        else:
            train = Solver(train_loader, None, None, args)
        train.train()   
    elif args.model == 'test':
        test_loader = voc_loader(args.root, 'test', None, args.mean_bgr, False, False, False, False, False, args)
        test = Solver(None, None, test_loader, args)
        test.test()
    else:
        raise ValueError('mode is not available!!!')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--name', type=str, default='voc')
    parser.add_argument('--root', type=str, default='path/to/VOCdevkit')
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--scales', type=list, default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument('--split', type=str, default='train')
    

    args = parser.parse_args()
    main(args)
