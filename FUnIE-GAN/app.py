from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import numpy as np
from os.path import exists
from keras.models import model_from_json
## local libs
from utils.data_utils import preprocess, deprocess


app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Process the image with your deep learning model
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
    
    #load image
    inp_img = image.resize((256,256))
    im = prepreprocess(inp_img)
    im = preprocess(im)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)

    print("\nLoaded data and model")

    # generate enhanced image
    gen = funie_gan_generator.predict(im)
    gen_img = deprocess(gen)[0]

    # save output images
    out_img = Image.fromarray(gen_img)

    #Image.fromarray(gen_img).save("/home/amitabha/demos/FUnIE/demo1/data/output.jpg")

    input_image_data = image_to_base64(inp_img)
    output_image_data = image_to_base64(out_img)

    input = "data:image/jpeg;base64," + input_image_data
    output = "data:image/jpeg;base64," + output_image_data

    return render_template("display.html", input = input, output = output)

def prepreprocess(im):
    if im.mode=='L': 
        copy = np.zeros((256, 256, 3))
        copy[:, :, 0] = im
        copy[:, :, 1] = im
        copy[:, :, 2] = im
        im = copy
    return np.array(im).astype(np.float32)

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode('ascii')
    return base64_image

if __name__ == "__main__":
    app.run()
