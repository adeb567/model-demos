import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import read_and_resize, preprocess, deprocess

## test funie-gan
checkpoint_dir  = '/home/amitabha/demos/FUnIE/demo1/models/gen_p/'
model_name_by_epoch = "model_15320_" 
## test funie-gan-up
#checkpoint_dir  = 'models/gen_up/'
#model_name_by_epoch = "model_35442_" 

model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

img_path = "/home/amitabha/demos/FUnIE/demo1/data/input.jpg"
inp_img = read_and_resize(img_path, (256, 256))
im = preprocess(inp_img)
im = np.expand_dims(im, axis=0) # (1,256,256,3)

# generate enhanced image
gen = funie_gan_generator.predict(im)
gen_img = deprocess(gen)[0]

# save output images
Image.fromarray(gen_img).save("/home/amitabha/demos/FUnIE/demo1/data/output.jpg")

