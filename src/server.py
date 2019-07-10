#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
from flask import Flask, redirect, url_for, render_template, request, \
    abort, jsonify
from evaluater.predict import main
import json
import urllib.request
import shutil
import os

app = Flask('server')


@app.route('/query/<model>', methods=['POST'])
def query(model):

    global images

    if request.method == 'POST':
        images = request.json

        if images:

            shutil.rmtree('temp')
            os.mkdir('temp')
            for image in images:
                filename_w_ext = os.path.basename(image)
                try:
                    urllib.request.urlretrieve(image, 'temp/'+ filename_w_ext)
                except:
                    print('An exception occurred :' + image)

            if model == 'technical':
            	model = '/models/MobileNet/weights_mobilenet_technical_0.11.hdf5'
            else: 
            	model = '/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5'

            result = main('MobileNet',
                          model,
                          'temp', 
                          None)

            return jsonify(result)

        return jsonify({'error': 'Image is not available'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
