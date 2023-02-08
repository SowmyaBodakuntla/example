import os

import numpy as np
from flask import Flask, request, render_template
from keras_preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
path=os.getcwd()
print(path)
model = load_model('mymodel.h5')
# Classes of traffic signs
classes = {0: 'Normal',
           1: 'Abnormal',
           }

app = Flask(__name__)
UPLOAD_FOLDER = "images/"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global filepath, value
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if uploaded_file.filename != '':
        uploaded_file.save(UPLOAD_FOLDER + filename)
        filepath = os.path.realpath(UPLOAD_FOLDER + filename)

        img = image.load_img(filepath, target_size=(32, 32))  # load the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        values = np.argmax(model.predict(images, batch_size=32), axis=-1)  # predict the label for the image

        s = [str(i) for i in values]
        a = int("".join(s))
        value = "Predicted Traffic🚦Sign is: " + classes[a]

    return render_template('index.html', value_text=value)


if __name__ == "__main__":
    app.run(debug=True)
