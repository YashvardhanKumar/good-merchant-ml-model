from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, send_from_directory, url_for
import os
from flask import Flask, jsonify, render_template, request
from keras import models
from ML_Model import ML_Model_Good_Merchant as GMM
from flask_cors import CORS
proxy_url = 'http://localhost:3000'
app = Flask(__name__)
CORS(app)
model = models.load_model('ML_Model/savedmodel.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "secret-key"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/qimage', methods=['POST'])
def image_binary():
    if 'file' not in request.files:
        flash('No file part')
        print(request.files)
        return redirect(proxy_url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading or no URL added')
        return redirect(proxy_url)

    elif file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = 'static/uploads/' + filename
        image = GMM.process_image_binary(filepath)
        pred_text = GMM.predict_image(image, model)
        os.remove(filepath)
        return {'q':pred_text}
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        print('mai hu 3')
        return redirect(proxy_url)


@app.route('/qimageurl', methods=['POST'])
def image_url():
    path = request.form['url']
    if path:
        image = GMM.process_image_url(path)
    else:
        flash('No image selected for uploading or no URL added')
        return redirect('/searchimage')
    pred_text = GMM.predict_image(image, model)
    return {'q':pred_text}


@app.after_request  # blueprint can also be app~~
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == '__main__':
    # app.run(debug=True)
    osPort = os.getenv("PORT")
    if osPort == None:
        port = 5000
    else:
        port = int(osPort)
    app.run(host='0.0.0.0', port=port)
