import os
import argparse
import pickle
import copy
import numpy as np

import tensorflow.contrib.keras as K
import tensorflow as tf

from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import layers
from .callbacks import save_predictions, save_best_model
import FaceImageGenerator as FID

def mse(y_true, y_pred):
    '''
    '''
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip),1)
    skip = tf.to_float(skip)
    cost = tf.reduce_mean(tf.square(y_true-y_pred),1)
    return cost * skip


def train_model(args):

    pip = FID.image_pipeline.FACE_pipeline(
            histogram_normalization = args.normalization,
            rotation_range = args.rotate,
            width_shift_range = args.transform,
            height_shift_range = args.transform,
            gaussian_range = args.gaussian_range,
            zoom_range = args.zoom,
            random_flip = True,
            allignment_type = 'similarity',
            grayscale = False,
            output_size = [256, 256],
            face_size = [224],
            )

    def data_provider(list_dat, aug):

        out = []
        for dat in list_dat:
            gen = dat[0]
            img = next(gen['img'])

            lab = next(gen['lab'])
            if lab.ndim==3:
                lab = lab.argmax(2)

            lab = np.int8(lab)

            out.append(-np.ones_like(lab))


        while True:
            for i, dat  in enumerate(list_dat):
                for gen in dat:
                    lab_list = copy.copy(out)

                    img = next(gen['img'])
                    lab = next(gen['lab'])

                    if lab.ndim==3:lab = lab.argmax(2)

                    lab_list[i] = lab

                    img_pp, _, _  = pip.batch_transform(img, preprocessing=True, augmentation=aug)

                    # crop the center of the image to size [244 244]
                    img_pp = img_pp[:,16:-16,16:-16,:]

                    img_pp -= np.apply_over_axes(np.mean, img_pp, [1,2])

                    yield [img_pp], lab_list



    if args.trainingData=='all':
        TR = [
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/disfa_tr.h5', 10, padding=-1),
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/disfa_te.h5', 10, padding=-1)
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/pain_tr.h5', 10, padding=-1),
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/pain_te.h5', 10, padding=-1)
                # ],
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/fera2015_te.h5', 10, padding=-1),
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/fera2015_tr.h5', 10, padding=-1)
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/imdb_wiki_te.h5', 10, padding=-1),
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/imdb_wiki_tr.h5', 10, padding=-1)
                # ]
        ]

    if args.trainingData=='tr':
        TR = [
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/disfa_tr.h5', 10, padding=-1),
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/pain_tr.h5', 10, padding=-1),
                # ],
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/fera2015_tr.h5', 10, padding=-1),
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/imdb_wiki_tr.h5', 10, padding=-1)
                # ]
            ]


        TE = [
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/disfa_te.h5', 10, padding=-1)
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/pain_te.h5', 10, padding=-1)
                # ],
                [
                    FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/fera2015_te.h5', 10, padding=-1)
                ],
                # [
                    # FID.provider.flow_from_hdf5('/homes/rw2614/data/similarity_256_256/imdb_wiki_te.h5', 10, padding=-1)
                # ]
            ]


    GEN_TR_a  = data_provider(TR, True)
    GEN_TE_na = data_provider(TE, False)
    GEN_TR_na = data_provider(TR, False)

    X, Y = next(GEN_TR_a)

    base_net = applications.resnet50.ResNet50( weights = 'imagenet' )

    inp_0 = base_net.input
    Z = base_net.get_layer('flatten_1').output

    out, loss  = [], []
    for i, y in enumerate(Y):
        net = layers.Dense( y.shape[1] )(Z)
        out.append( net )
        loss.append(mse)

    model = K.models.Model([inp_0], out)
    model.summary()

    model.compile(
            optimizer = K.optimizers.Adadelta(
                lr = 1.,
                rho = 0.95, 
                epsilon = 1e-08, 
                decay = 1e-5,
                ),
            loss = loss 
            )

    # K.utils.plot_model(model, to_file=args.log_dir+'/model.png', show_shapes=True)

    model.fit_generator(
            generator = GEN_TR_a, 
            steps_per_epoch = 2000,
            epochs = args.epochs, 
            max_q_size = 100,
            validation_data = GEN_TE_na,
            validation_steps = 100,
            callbacks=[
                save_predictions(GEN_TR_na, args.log_dir+'/TR_'),
                save_predictions(GEN_TE_na, args.log_dir+'/TE_'),
                save_best_model(GEN_TE_na,  args.log_dir),
                K.callbacks.ModelCheckpoint(args.log_dir+'/model.h5',save_weights_only=True)
                ]
            )

    return model, pip

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train model for AU intenistie estimation')
    parser.add_argument("-tr","--trainingData", type=str, default='tr')
    parser.add_argument("-l","--log_dir", type=str, default='/tmp')

    # parser.add_argument("-r","--rotate", type=float, default=20)
    # parser.add_argument("-e","--epochs", type=int, default=100)
    # parser.add_argument("-g","--gaussian_range", type=float, default=4)
    # parser.add_argument("-n","--normalization", type=int, default=0)
    # parser.add_argument("-t","--transform", type=float, default=0.2)
    # parser.add_argument("-z","--zoom", type=float, default=0.3)

    # parser.add_argument("-r","--rotate", type=float, default=9)
    # parser.add_argument("-e","--epochs", type=int, default=100)
    # parser.add_argument("-g","--gaussian_range", type=float, default=2)
    # parser.add_argument("-n","--normalization", type=int, default=0)
    # parser.add_argument("-t","--transform", type=float, default=0.05)
    # parser.add_argument("-z","--zoom", type=float, default=0.1)

    parser.add_argument("-r","--rotate", type=float, default=0)
    parser.add_argument("-e","--epochs", type=int, default=100)
    parser.add_argument("-g","--gaussian_range", type=float, default=0)
    parser.add_argument("-n","--normalization", type=int, default=0)
    parser.add_argument("-t","--transform", type=float, default=0)
    parser.add_argument("-z","--zoom", type=float, default=0)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    pickle.dump(args,open(args.log_dir+'/args.pkl','wb'))

    train_model(args)
