import os
import sys
from time import time

import cv2
import numpy as np
from flask import Flask, request, jsonify

from main import process_image

""" ROUTES """
app = Flask(__name__)
fen = ""


@app.route('/')
def hello():
    return 'Chess ID. usage: /upload'


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        fen = ""
        if file and allowed_file(file.filename):
            start = time()
            img = np.asarray(bytearray(file.read()))
            tmp_path = os.path.join("tmp/", file.filename)
            cv2.imwrite(tmp_path, img)
            print("Wrote image to ", tmp_path)
            try:
                fen = process_image(tmp_path)
                print("Success")
            except:
                print("ERROR no fen found")
                print("Unexpected error:", sys.exc_info()[0])
        json = {'fen': fen, 'time': time() - start}
        return jsonify(json)
    return '''
    <!doctype html>
    <title>Chess ID</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
