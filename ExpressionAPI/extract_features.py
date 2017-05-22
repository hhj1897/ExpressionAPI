import os
import numpy as np
import math
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import applications
import FaceImageGenerator as FIG
from skimage.io import imread


class Feature_Extractor():

    def __init__(self, weights='default', use_GPU=True, batch_size=10, verbose=0):

        if not use_GPU:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if weights== 'default':
            weights= os.path.dirname(__file__)+'/models/ResNet50_aug_1.1/best_model.h5'

        base_net = applications.resnet50.ResNet50( weights = None )
        inp = base_net.input

        net = base_net.get_layer('flatten_1').output

        self.model = K.models.Model(inp, net)

        self.model.load_weights(weights, by_name=True)
        self.batch_size = batch_size
        self.verbose = verbose

        self.preprocess = FIG.image_pipeline.FACE_pipeline(
            histogram_normalization = False,
            output_size = [256, 256],
            face_size= 224,
            allignment_type = 'similarity'
            )

    def get_input_features_from_numpy(self, img_batch):
        img, pts, pts_raw = self.preprocess.batch_transform(img_batch, face_detect=True, preprocessing=False)
        img = np.float32(img)
        img_pp = img[:,16:-16,16:-16,:]
        img_pp -= np.apply_over_axes(np.mean, img_pp, [1,2])
        return img_pp, pts, pts_raw

    def get_face_features_from_numpy(self, img_batch):
        img, _, _ =  self.get_input_features_from_numpy(img_batch)
        Z = self.model.predict(img, batch_size=self.batch_size)
        return Z

    def get_all_features_from_numpy(self, img_batch):
        img, pts, pts_raw =  self.get_input_features_from_numpy(img_batch)
        Z = self.model.predict(img, batch_size=self.batch_size)
        return img, pts, pts_raw, Z

    def get_face_features_from_files(self, list_files):
        list_files = np.array(list_files)
        def img_generator(list_files):
            t0, t1  = 0, self.batch_size
            while True:
                img_batch = [imread(f) for f in list_files[t0:t1]]

                # bug/featur of keras: 
                # in if end of generator is reached, return single frame to prevent crash of predict_generator 
                if len(img_batch)==0:img_batch = [imread(list_files[0])]

                img, _, _ = self.get_input_features_from_numpy(img_batch)
                t0 += self.batch_size
                t1 += self.batch_size
                yield img

        gen = img_generator(list_files)
        nb_batches = math.ceil((len(list_files))/self.batch_size)
        out = self.model.predict_generator(gen, nb_batches, max_q_size=10, workers=1)
        return out

    def get_all_features_from_files(self, list_files):
        list_files = np.array(list_files)
        def img_generator(list_files):
            t0, t1  = 0, self.batch_size
            while True:
                img_batch = [imread(f) for f in list_files[t0:t1]]

                # bug/featur of keras: 
                # in if end of generator is reached, return single frame to prevent crash of predict_generator 
                if len(img_batch)==0:img_batch = [imread(list_files[0])]

                img, pts, pts_raw = self.get_input_features_from_numpy(img_batch)
                t0 += self.batch_size
                t1 += self.batch_size
                yield img, pts, pts_raw

        gen = img_generator(list_files)

        IMG, PTS, PTS_RAW, Z = [], [], [], []
        n_batches = math.ceil((len(list_files))/self.batch_size)
        for i in range(n_batches):
            if self.verbose==1:print('batch:',i,'/',n_batches)
            img_batch, pts_batch, pts_raw_batch = next(gen)
            z = self.model.predict(img_batch)
            IMG.extend(img_batch)
            PTS.extend(pts_batch)
            PTS_RAW.extend(pts_raw_batch)
            Z.extend(z)

        return np.stack(IMG), np.stack(PTS), np.stack(PTS_RAW), np.stack(Z)
