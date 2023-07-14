from flask import Flask, request, jsonify
import feedparser
import langid
from keybert import KeyBERT
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
import math
import requests
import torch
import nltk
import os

#nltk.download('punkt')
os.environ['CURL_CA_BUNDLE'] = ''

requests.Session().verify = False

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

# TODO execute before first request and load models
def init_models():
    print('hi')


def create_app():
    app = Flask(__name__)
    init_models()

    @app.route("/")
    def hello_world():
        # TODO welche uuids muss man tätigen
        return "Help Information Incomming"

    @app.route('/<uuid>', methods=['GET', 'POST'])
    def add_message(uuid):
        content_type = request.headers.get('Content-Type')
        content_check = check_content_type(content_type)
        data = None
        if content_check:
            data = request.get_json()

        if uuid == 'rss':
            return get_rss()
        elif uuid == 'detectLng':
            if data is not None:
                return get_language(data)
            else:
                return 'error'
        elif uuid == 'sentiment':
            if data is not None:
                return get_sentiment(data)
            else:
                return 'error'
        elif uuid == 'summarize':
            if data is not None:
                return get_summary(data)
            else:
                return 'error'
        elif uuid == 'keywords':
            if data is not None:
                return get_keywords(data)
            else:
                return 'error'
        elif uuid == 'translate':
            if data is not None:
                return get_translation(data)
            else:
                return 'error'
        else:
            return 'error'

    return app


def check_content_type(content_type):
    if content_type == 'application/json':
        return True
    else:
        return False


def get_rss():
    rss_url = 'https://www.heise.de/rss/heise.rdf'
    feed = feedparser.parse(rss_url).entries
    # return json.dumps(feed)
    return feed


def get_keywords(body):
    input = body.get('input')
    kw_model = KeyBERT()
    body['output'] = keywords = kw_model.extract_keywords(input)
    return jsonify(body)


def get_summary(body):
    input = body.get('input')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    body['output'] = summarizer(input, max_length=130, min_length=30, do_sample=False)
    return jsonify(body)


def get_sentiment(body):
    input = body.get('input')
    classifier = pipeline("sentiment-analysis")
    body['output'] = classifier(input)
    return jsonify(body)


def get_language(body):
    input = body.get('input')
    detected_language, confidence = langid.classify(input)
    body['output'] = detected_language, confidence
    return jsonify(body)


def get_translation(body):
    translation_en_de = pipeline("translation_en_to_de")
    input = body.get('input')
    mname = 'Helsinki-NLP/opus-mt-de-en'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    model.to(device)

    if input:
        if body['action'] == 'translate_en_de':
            output = translation_en_de(input, max_length=512)
            body['output'] = output[0]['translation_text']
        if body['action'] == 'translate_de_en':
            lt = LineTokenizer()
            batch_size = 8
            paragraphs = lt.tokenize(input)
            translated_paragraphs = []

            for paragraph in paragraphs:
                sentences = sent_tokenize(paragraph)
                batches = math.ceil(len(sentences) / batch_size)
                translated = []
                for i in range(batches):
                    sent_batch = sentences[i * batch_size:(i + 1) * batch_size]
                    model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        translated_batch = model.generate(**model_inputs)
                    translated += translated_batch
                translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                translated_paragraphs += [" ".join(translated)]
            translated_text = "\n".join(translated_paragraphs)
            body['output'] = translated_text
    return jsonify(body)


if __name__ == '__main__':
    app = create_app()
    app.run()
