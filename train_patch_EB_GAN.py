#coding:utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
from chainer.datasets import tuple_dataset#no learning
import six
import os
import SimpleITK as sitk

from chainer import cuda, optimizers, serializers, Variable, iterators
from chainer import training
from chainer.training import extensions

import argparse
import sys
from tqdm import tqdm
import datetime


#import unet
import AE

#add
import myUpdater
import ioFunction_version_4_3 as IO
from Augumentation import Augmentor as ag

# Load the datasets(n-classes)
def load_image(filepath, rootpath, patchsize, label_num):
    file_name = []
    with open(filepath) as f:
        all_line = f.readlines()
        for line in all_line:
            file_name.append(line.replace("\n",""))
    
    tmp = np.zeros((patchsize, patchsize), dtype = np.float32)
    # input images
    x = np.zeros((len(file_name), 1, patchsize, patchsize), dtype = np.float32)
    # supervised data (label)
    t = np.zeros((len(file_name), patchsize, patchsize), dtype = np.int32)
    with tqdm(total=len(file_name)) as pbar:
        for i in range(len(file_name)):
            
            img, d_ = IO.read_mhd_and_raw_withoutSitk(rootpath + "/image/" + file_name[i] + ".mhd")
            #nda_img = img.reshape((d_['DimSize'][1], d_['DimSize'][0])).astype(np.float32) / 255  # img => [0,1]
            nda_img = img.reshape((d_['DimSize'][1], d_['DimSize'][0])).astype(np.float32) # img => zve = 0, var = 1
            label, d_ = IO.read_mhd_and_raw_withoutSitk(rootpath + "/label/" + file_name[i] + ".mhd")
            nda_label = label.reshape((d_['DimSize'][1], d_['DimSize'][0])).astype(np.int32)

            #img = sitk.ReadImage(rootpath + "/image/" + file_name[i] + ".mhd")
            #nda_img = sitk.GetArrayFromImage(img).astype(np.float32)  # img => zve = 0, var = 1
            #label = sitk.ReadImage(rootpath + "/label/" + file_name[i] + ".mhd")
            #nda_label = sitk.GetArrayFromImage(label).astype(np.int32)

            
            if label_num == 2:
                # train target => bkg, accumulate
                nda_label[np.where(nda_label == 2)] = -1
                nda_label[np.where(nda_label == 4)] = 1
            elif label_num == 3:
                # train target => bkg, normal, abnormal
                nda_label[np.where(nda_label == 2)] = -1
                nda_label[np.where(nda_label == 4)] = 2
            elif label_num == 4:
                # train target => bkg, excluded, normal, abnormal
                nda_label[np.where(nda_label == 4)] = 3

            # input
            x[i,0,:,:] = nda_img
            # label
            t[i,:,:] = nda_label
            pbar.update(1)
    temp = tuple_dataset.TupleDataset(x, t)
    return temp













def main():

    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=2,
                        help='Number of images in each mini-batch')
    parser.add_argument('--patchsizex', '-px', type=int, default=132,
                        help='Size of images in each patch')
    parser.add_argument('--patchsizey', '-py', type=int, default=132,
                        help='Size of images in each patch')
    parser.add_argument('--patchsizez', '-pz', type=int, default=100,
                        help='Size of images in each patch')
    parser.add_argument('--margin', '-m' , type=int, default=44,
                        help='difference of one side between input and output')                        

    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of image files')
    parser.add_argument('--dataset', '-i', default=os.path.dirname(os.path.abspath(__file__)) + "/data/",
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default=os.path.dirname(os.path.abspath(__file__)) + "/result/patch_based",
                        help='Directory to output the result')
    parser.add_argument('--data', '-d', type=str, default="norm",
                        help='input image data')
    parser.add_argument('--optimizer', '-opt', type=str, default="Adam",
                        help='Optimizer type')

    parser.add_argument('--network', '-n', type=str, default="cars",
                        help='Network archtecture')

    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--labelnum', '-l', type=int, default=4,
                        help='Output classes number')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--initial_lr', type=float, default=0.01,
                        help='Initial of learning rate')
    args = parser.parse_args()


    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))
    print('# patchsize: {}'.format(args.patchsizex))
    print('# patchsize: {}'.format(args.patchsizey))
    print('# patchsize: {}'.format(args.patchsizez))
    print('# Classes: {}'.format(args.labelnum))
    print('# Data: {}'.format(args.data))
    print('# Optimizer: {}'.format(args.optimizer))
    print('# Network: {}'.format(args.network))
    print('')

    today = datetime.date.today()
    
    if args.network == "unet":
        network = "unet"
    elif args.network == "resunet":
        network = "unet_with_resblock"
    elif args.network == "unet_kai":
        network = "unet_kai"
    elif args.network == "resunet_kai":
        network = "unet_with_resblock_kai"
    elif args.network == "resunet_kaikai":
        network = "unet_with_resblock_kai_kai"
    elif args.network == "cars":
        network = "unet_with_resblock_CARS"
    elif args.network == "cars_do":
        network = "unet_with_resblock_CARS_dropout"
    elif args.network == "without":
        network = "unet_without_resblock_CARS"

    data_mode = args.data
    if data_mode == "zscore":
        data_path = args.dataset + str(args.patchsize) + "/zscore"
    elif data_mode == "norm":
        data_path = args.dataset + str(args.patchsize) + "/Value_Normalized"
    elif data_mode == "aug":
        data_path = args.dataset + str(args.patchsize) + "/Data_Augmentation_VN"
    elif data_mode == "aug_rot":
        data_path = args.dataset + str(args.patchsize) + "/Data_Augmentation_VN_rotated"
    elif data_mode == "aug_flip":
        data_path = args.dataset + str(args.patchsize) + "/Data_Augmentation_VN_flip"

    label_num = args.labelnum
    
    out_path = args.out + "/" + str(args.patchsize) + "/" + network + "/" + args.optimizer  + "/" + args.data + "/temp/" + str(today) + "_" + str(args.iteration) + "itr_" + str(args.batchsize) + "bs_class" + str(label_num)
    print("# Output Path:",out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    print('')

    # Dataset
    print("------Loading the Dataset------")
    print("train_data.txt")
    train = load_image(os.path.join(data_path,"train_data.txt"), data_path, args.patchsize, label_num)
    print("validation_data.txt")
    val = load_image(os.path.join(data_path,"validation_data.txt"), data_path, args.patchsize, label_num)

    print("------Loaded the Dataset------")
    print("train_data num : ",len(train))
    print("val_data num : ",len(val))

    # Iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

    # Model
    if args.network == "unet":
        model = unet.UNet2D(label_num)
    elif args.network == "resunet":
        model = unet_with_resblock.ResUNet2D(label_num)
    elif args.network == "unet_kai":
        model = unet_kai.UNet2D(label_num)
    elif args.network == "resunet_kai":
        model = unet_with_resblock_kai.ResUNet2D(label_num)
    elif args.network == "resunet_kaikai":
        model = unet_with_resblock_kai_kai.ResUNet2D(label_num)
    elif args.network == "cars":
        model = unet_with_resblock_CARS.ResUNet2D(label_num)
    elif args.network == "cars_do":
        model = unet_with_resblock_CARS_dropout.ResUNet2D(label_num)
    elif args.network == "without":
        model = unet_without_resblock_CARS.ResUNet2D(label_num)
    
    #serializers.load_npz("result/model_iter_10000", model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU


    # Optimizer
    if args.optimizer == "MSGD":
        optimizer = optimizers.MomentumSGD(lr=args.initial_lr)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    elif args.optimizer == "Adam":
        optimizer = optimizers.Adam()
        optimizer.setup(model)


    # Updater
    #updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    updater = myUpdater.UNet2D_Updater(model=model,iterator=train_iter,
        optimizer={"model":optimizer}, device=args.gpu)

    # Trainer
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=out_path)

    # ------------------------trainer setup------------------------ #
    # Triggers
    log_trigger = (50, 'iteration')
    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval_2 = (args.snapshot_interval*10, 'iteration')

    # Save model for trainer
    trainer.extend(extensions.snapshot
        (filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval_2)
    # Save model for model
    trainer.extend(extensions.snapshot_object(
        model, filename='iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)

    # Write a log of evaluation statistics for each 100 iterations
    trainer.extend(extensions.LogReport(trigger=log_trigger))

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'model/loss', 'model/val/loss', 'model/accuracy', 'model/val/accuracy']), trigger=log_trigger)

    # Evaluate the model with the val dataset for each epoch
    trainer.extend(myUpdater.UNet2D_Evaluator(val_iter, model, device=args.gpu), trigger=log_trigger)

    # Dump a computational graph from 'loss' variable at the first iteration
    trainer.extend(extensions.dump_graph('model/loss'), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('model/accuracy'), trigger=log_trigger)

    # Save the loss plot data (image)
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['model/loss', 'model/val/loss'], 'iteration', file_name='loss.png', trigger=log_trigger))
        trainer.extend(extensions.PlotReport(
            ['model/accuracy', 'model/val/accuracy'], 'iteration', file_name='accuracy.png', trigger=log_trigger))

    # Print a progress bar
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()

    # Save the trained model
    model.to_cpu() # converting to cpu model
    serializers.save_npz(str(out_path) + "/mymodel.npz", model) # save as npz
    
    if args.resume:
        # Resume from a snapshot
        serializers.load_npz(args.resume, trainer)

if __name__ == '__main__':
    main()
