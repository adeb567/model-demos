# model-demos
Steps for Individual Apps:

- Setup the Flask environment using: https://phoenixnap.com/kb/install-flask

- Set the model path in app.py (For SVAM - download the pretrained model weights from https://drive.google.com/drive/folders/19OyCuvZ0sLlHdmF1TrDRa4QrFLwhCWRk)

- Run the app using 'python3 app.py'

Steps for Integrated App:

- Setup the Flask environment using: https://phoenixnap.com/kb/install-flask

- Set the model paths in app.py in Intregated/demo*

  (For SVAM - download the pretrained model weights from https://drive.google.com/drive/folders/19OyCuvZ0sLlHdmF1TrDRa4QrFLwhCWRk and place it in Intregated/demo3/models)

  Intregated/demo1 - FUnIE
  
  Intregated/demo2 - Deep_SESR
  
  Intregated/demo3 - SVAM

- Run the app using 'python3 app.py' in /Integrated

  flask run --host=0.0.0.0

# Deployment
https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04

gunicorn --bind 0.0.0.0:5000 wsgi:app

# Project Links
FUnIE-GAN - https://irvlab.cs.umn.edu/projects/funie-gan

Deep-SESR - https://irvlab.cs.umn.edu/projects/deep-sesr

SVAM - https://irvlab.cs.umn.edu/visual-attention-modeling/svam
