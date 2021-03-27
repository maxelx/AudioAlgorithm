import random
import os
from flask import Flask, request, jsonify
from AudioAlgorithm import percentageAudio
from Algorithm import algorithm

app = Flask(__name__)

@app.route("/",methods=["GET"])
def func():
    return "ciao"

@app.route("/predict", methods=["POST"])
def predict():
    #get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    #invoke model
    #model = percentageAudio(file_name)
    mfcc = algorithm(file_name)
    mfcc2 = percentageAudio(file_name)

    #remove the audio file stored
    os.remove(file_name)

    #send back the prediction in json format
    result = (mfcc + mfcc2)/2
    data = {"Result": result}

    #print("Result", data)
    return jsonify(data)

if __name__ == "__main__" :
    #app.run(debug=False, host= '0.0.0.0')
    app.run(port=5000, debug=False)
