from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from PIL import Image
import numpy as np
import base64
import os

#from keras.models import model_from_json
#model = model_from_json(open('C:/Users/Ryunosuke Ichiyasu/Desktop/Guitar_Chords_Classification_Project/Guitar_Chords_Classification_Project/Guitar_Chords_Classification/model.json').read())
#model.road_weights('C:/Users/Ryunosuke Ichiyasu/Desktop/Guitar_Chords_Classification_Project/Guitar_Chords_Classification_Project/Guitar_Chords_Classification/weights.h5')
from keras.models import load_model
model = load_model('C:/Users/Ryunosuke Ichiyasu/Desktop/Guitar_Chords_Classification_Project/Guitar_Chords_Classification_Project/Guitar_Chords_Classification/model_django.h5', compile=False)

#ex = np.zeros(150, 150)
#ex = np.expand_dims(ex, axis=0)
#model.predict(image)
import tensorflow as tf
graph = tf.get_default_graph()

def GuitarChords(image):
    with graph.as_default():
        preds = model.predict(image)
        #pred_argmax = np.argmax(preds[0])
        return preds[0]

def upload(request):

    files = request.FILES.getlist("files[]")

    for memory_file in files:
        root, ext = os.path.splitext(memory_file.name)

        if ext != '.jpg':
            message = "[ERROR] The file format is not JPG"
            return render(request, 'Guitar_Chords_Classification/index.html', {
                "message": message,
                })

    if request.method == 'POST' and files:
        result = []
        chord_list = []
        preds_list = []

        for file in files:
            image = Image.open(file)
            image = image.resize((150, 150))
            image = np.array(image)
            image = image.astype("float32")
            image = image / 255
            image = np.expand_dims(image, axis=0)
            preds = GuitarChords(image)

            pred_argmax = np.argmax(preds)
            if pred_argmax == 0:
                chord = 'C'
            elif pred_argmax == 1:
                chord = 'Dm'
            elif pred_argmax == 2:
                chord = 'Em'
            elif pred_argmax == 3:
                chord = 'F'
            elif pred_argmax == 4:
                chord = 'G'
            elif pred_argmax == 5:
                chord = 'Am'
            elif pred_argmax == 6:
                chord = 'Bm'

            chord_list.append(chord)

            preds = preds * 100
            preds = np.round(preds, 1)
            print(preds)

            preds_list.append(preds)

        for file, chord, preds in zip(files, chord_list, preds_list):
            file.seek(0)
            file_name = file
            src = base64.b64encode(file.read())
            src = str(src)[2:-1]
            result.append((src, chord, preds))

        context = {
                'result' : result
            }

        return render(request, 'Guitar_Chords_Classification/result.html', context)

    else:
        return redirect('index')
