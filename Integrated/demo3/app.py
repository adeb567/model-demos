from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import os
import numpy as np
from os.path import exists

## local libs
from demo3.models.svam_model import SVAM_Net
from demo3.utils.data_utils import preprocess, deprocess_mask

from flask import Blueprint

app3_bp = Blueprint('app3', __name__, static_url_path='/static')

@app3_bp.route('/app3')
def app3_route():
    return render_template("index3.html")

@app3_bp.route("/app3/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Process the image with your deep learning model
    ## svam
    # input and output data shape
    im_res, chans = (256, 256), 3
    im_shape = (256, 256, 3)

    ## load specific model 
    model_h5 = "/home/amitabha/demos/Integrated/demo3/models/SVAM_Net.h5"
    assert os.path.exists(model_h5), "h5 model not found"
    model = SVAM_Net()
    model.load_weights(model_h5)
    print("\nLoaded model")

    # prepare data
    inp_img = image.resize(im_res)
    im = np.expand_dims(preprocess(np.array(inp_img)), axis=0)        

    # generate saliency
    saml, sambu, samd, out = model.predict(im)
    _, out_bu, _, out_tdr = deprocess_gens(saml, sambu, samd, out, im_res)

    # save output images
    out_img1 = Image.fromarray(out_bu)
    out_img2 = Image.fromarray(out_tdr)

    input_image_data = image_to_base64(inp_img)
    output_image_data1 = image_to_base64(out_img1)
    output_image_data2 = image_to_base64(out_img2)

    input = "data:image/jpeg;base64," + input_image_data
    output1 = "data:image/jpeg;base64," + output_image_data1
    output2 = "data:image/jpeg;base64," + output_image_data2

    return render_template("display3.html", input = input, output1 = output1, output2 = output2)



def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode('ascii')
    return base64_image

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

