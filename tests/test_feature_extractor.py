from skimage.io import imread
import glob
import os
import numpy as np

import ExpressionAPI 

pwd = os.path.dirname(os.path.abspath(__file__))


img_files = sorted(list(glob.glob(pwd+'/test_images/S148_002/*.png')))
img_batch = [imread(f) for f in img_files]




FE = ExpressionAPI.Feature_Extractor()


# extract face features from list of jpg/png files
Z = FE.get_face_features_from_files(img_files)

# check if shape is correct
assert( Z.shape==(15,2048) )

# check if features are valid
assert( np.all(np.isnan(Z)==False) )




# extract all possible features from list of numpy arrays with shape [X,Y,C]
img, pts, pts_raw, Z  = FE.get_all_features_from_numpy(img_batch)

# check if shape is correct
assert( Z.shape==(15,2048) )
assert( img.shape==(15,224, 224, 3) )
assert( pts.shape==(15,68, 2) )
assert( pts_raw.shape==(15,68, 2) )

# check if features are valid
assert( np.all(np.isnan(Z)==False) )
assert( np.all(np.isnan(img)==False) )
assert( np.all(np.isnan(pts)==False) )
assert( np.all(np.isnan(pts_raw)==False) )





# extract low level features from list of numpy arrays with shape [X,Y,C]
img, pts, pts_raw = FE.get_input_features_from_numpy(img_batch)

# check if shape is correct
assert( img.shape==(15,224, 224, 3) )
assert( pts.shape==(15,68, 2) )
assert( pts_raw.shape==(15,68, 2) )

# check if features are valid
assert( np.all(np.isnan(img)==False) )
assert( np.all(np.isnan(pts)==False) )
assert( np.all(np.isnan(pts_raw)==False) )


print('_tests_successful__')
