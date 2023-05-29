import numpy as np
from glob import glob
from ntpath import basename
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import preprocess, deprocess
from utils.data_utils import deprocess_uint8, deprocess_mask


# input and output data shape
scale = 2 
hr_w, hr_h = 640, 480 # HR
lr_w, lr_h = 320, 240 # LR (1/2x)
lr_res, lr_shape = (lr_w, lr_h), (lr_h, lr_w, 3)
hr_res, hr_shape = (hr_w, hr_h), (hr_h, hr_w, 3)

## test deep-sesr
checkpoint_dir  = '/home/amitabha/demos/Deep-SESR/demo2/models/'
ckpt_name =  "deep_sesr_2x_1d"

model_h5 = checkpoint_dir + ckpt_name + ".h5"  
model_json = checkpoint_dir + ckpt_name + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
generator = model_from_json(loaded_model_json)
# load weights into new model
generator.load_weights(model_h5)
print("\nLoaded data and model")


# prepare data
img_path = "/home/amitabha/demos/Deep-SESR/demo2/data/4.jpg"
img_name = basename(img_path).split('.')[0]
img_lrd = np.array(Image.open(img_path).resize(lr_res))
im = np.expand_dims(preprocess(img_lrd), axis=0)

# get output
gen_op = generator.predict(im)
gen_lr, gen_hr, gen_mask = gen_op[0], gen_op[1], gen_op[2]

# process raw outputs 
gen_lr = deprocess_uint8(gen_lr).reshape(lr_shape)
gen_hr = deprocess_uint8(gen_hr).reshape(hr_shape)
gen_mask = deprocess_mask(gen_mask).reshape(lr_h, lr_w)

# save generated images
samples_dir = '/home/amitabha/demos/Deep-SESR/demo2/data/'
Image.fromarray(img_lrd).save(join(samples_dir, img_name+'.png'))
Image.fromarray(gen_lr).save(join(samples_dir, img_name+'_En.png'))
Image.fromarray(gen_mask).save(join(samples_dir, img_name+'_Sal.png'))
Image.fromarray(gen_hr).save(join(samples_dir, img_name+'_SESR.png'))
