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