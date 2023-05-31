import os
import numpy as np
from PIL import Image
import keras
## local libs
from models.svam_model import SVAM_Net
from utils.data_utils import preprocess, deprocess_mask

## test svam
def sigmoid(x):
    """ Numerically stable sigmoid
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x))
                   )

def deprocess_gens(saml, sambu, samtd, out, im_res):
    """ Post-process all outputs
    """
    samtd, sambu =  sigmoid(samtd), sigmoid(sambu)
    out = deprocess_mask(out).reshape(im_res) 
    saml = deprocess_mask(saml).reshape(im_res) 
    samtd = deprocess_mask(samtd).reshape(im_res) 
    sambu = deprocess_mask(sambu).reshape(im_res)
    return saml, sambu, samtd, out


# input and output data shape
im_res, chans = (256, 256), 3
im_shape = (256, 256, 3)

## load specific model 
model_h5 = "/home/amitabha/demos/SVAM/demo3/models/SVAM_Net.h5"
assert os.path.exists(model_h5), "h5 model not found"
model = SVAM_Net()
model.load_weights(model_h5)
print("\nLoaded model")

# prepare data
img_path = "/home/amitabha/demos/SVAM/demo3/data/input.jpg"
inp_img = np.array(Image.open(img_path).resize(im_res))
im = np.expand_dims(preprocess(inp_img), axis=0)        

# generate saliency
saml, sambu, samd, out = model.predict(im)
_, out_bu, _, out_tdr = deprocess_gens(saml, sambu, samd, out, im_res)

# save output images
Image.fromarray(out_bu).save("/home/amitabha/demos/SVAM/demo3/data/output_bu.png")
Image.fromarray(out_tdr).save("/home/amitabha/demos/SVAM/demo3/data/output_tdr.png")
