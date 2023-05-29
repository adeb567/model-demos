from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import numpy as np
from os.path import exists
from keras.models import model_from_json
## local libs
from utils.data_utils import preprocess, deprocess
from utils.data_utils import deprocess_uint8, deprocess_mask


app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Process the image with your deep learning model
   
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
    img_lrd = np.array(image.resize(lr_res))
    im = np.expand_dims(preprocess(img_lrd), axis=0)

    # get output
    gen_op = generator.predict(im)
    gen_lr, gen_hr, gen_mask = gen_op[0], gen_op[1], gen_op[2]

    # process raw outputs 
    gen_lr = deprocess_uint8(gen_lr).reshape(lr_shape)
    gen_hr = deprocess_uint8(gen_hr).reshape(hr_shape)
    gen_mask = deprocess_mask(gen_mask).reshape(lr_h, lr_w)

    # save generated images
    #samples_dir = '/home/amitabha/demos/Deep-SESR/demo2/data/'

    input = Image.fromarray(img_lrd)
    output_en = Image.fromarray(gen_lr)
    output_sal = Image.fromarray(gen_mask)
    output_sesr = Image.fromarray(gen_hr)


    input_image_data = image_to_base64(input)
    output_enhanced = image_to_base64(output_en)
    output_saliency = image_to_base64(output_sal)
    output_super = image_to_base64(output_sesr)

    input = "data:image/png;base64," + input_image_data
    output1 = "data:image/png;base64," + output_enhanced
    output2 = "data:image/png;base64," + output_saliency
    output3 = "data:image/png;base64," + output_super


    return render_template("display.html", input = input, output1 = output1, output2 = output2, output3 = output3)


def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode('ascii')
    return base64_image

if __name__ == "__main__":
    app.run()
