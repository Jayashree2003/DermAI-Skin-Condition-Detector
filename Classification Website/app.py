import os
import PIL
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file, send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import normalize
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
vgg19_model = load_model('models/vgg19.h5')
MobileNetV3Small_model = load_model('models/mobilenetv3small.h5')
resnet152_model = load_model('models/resnet152.h5')
fusion_model = load_model('models/prediction_model.h5')

app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')
@app.route('/pred', methods=['GET'])
def pred():
    return render_template('predict.html')
@app.route('/classes', methods=['GET'])
def classes():
    return render_template('classes.html')
@app.route('/prevent', methods=['GET'])
def prevent():
    return render_template('prevention.html')
@app.route('/team', methods=['GET'])
def team():
    return render_template('team.html')
@app.route('/', methods=['POST'])
def predict():
    predictions = []
    pred=[]
    imagefiles = request.files.getlist('imagefile')
    for imagefile in imagefiles:
        image_path = "static/uploads" + imagefile.filename
        imagefile.save(image_path)
        img = load_img(image_path, target_size=(112, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images = img_array[:, :, :, ::-1]
        m0 = np.mean(images[:, :, :, 0])
        m1 = np.mean(images[:, :, :, 1])
        m2 = np.mean(images[:, :, :, 2])
        images[:, :, :, 0] -= m0
        images[:, :, :, 1] -= m1
        images[:, :, :, 2] -= m2
        vgg19_features = vgg19_model.predict(images)
        MobileNetV3Small_features = MobileNetV3Small_model.predict(images)
        resnet_features = resnet152_model.predict(images)
        F_vgg_normalized = normalize(vgg19_features, norm='l2', axis=1)
        MobileNetV3Small_normalized = normalize(MobileNetV3Small_features, norm='l2', axis=1)
        F_resnet_normalized = normalize(resnet_features, norm='l2', axis=1)
        prediction = fusion_model.predict([F_vgg_normalized, MobileNetV3Small_normalized, F_resnet_normalized])
        predicted_class = np.argmax(prediction)
        class_names = ['Actinic Keratosis', 'Basal cell carcinoma', 'Benign keratosis lesions', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
        final_class=class_names[predicted_class]
        classification = '%s' % (final_class)
        image_rgb = img.convert('RGB')
        buffered = BytesIO()
        image_rgb.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        predictions.append({'Image Name': imagefile.filename, 'Prediction': classification,'Image Base64': image_base64})
        pred.append({'Image Name': imagefile.filename, 'Prediction': classification})

    df = pd.DataFrame(pred)
    df.to_csv('./predictions.csv', index=False)
    return render_template('predict.html', predictions=predictions)


@app.route('/download_csv', methods=['GET'])
def download_csv():
    file_path = './predictions.csv'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "CSV file not found.", 404

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
