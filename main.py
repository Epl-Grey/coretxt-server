import cv2
import pytesseract
from flask import Flask, request
from llama_cpp import Llama
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1000 * 1000

# llm = Llama(model_path="/home/artem/llama.cpp/models/llama-7B/ggml-model.bin")


# def set_punctuation(text: str):
#     prompt = f"Расставь знаки препинания в предложении: '{text}'"
#     output = llm(prompt, max_tokens=32, stop=["Q:", "\n"], echo=True)
#     return output



@app.route("/")
def index():
    return "This is backend for project on hackaton"


@app.post('/text')
def text_post():
    print(request.form)
    text = request.form['text']
    return text


@app.post('/image')
def image_post():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 128, cv2.THRESH_OTSU)[1]
    recognized_text = pytesseract.image_to_string(thresh, lang="rus")
    print("Recognized: " + recognized_text)
    return recognized_text if recognized_text != "" else "Not recognized."


if __name__ == '__main__':
    app.run("0.0.0.0", 8000)
