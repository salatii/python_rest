from flask import Flask, request, jsonify
import feedparser
import langid
from keybert import KeyBERT
from transformers import pipeline
import requests
import os
from tasks import translation

#nltk.download('punkt')
os.environ['CURL_CA_BUNDLE'] = ''

requests.Session().verify = False

# TODO execute before first request and load models
def init_models():
    translation.init()


def create_app():
    app = Flask(__name__)
    init_models()

    @app.route("/")
    def hello_world():
        # TODO
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
    action = body['action']
    input = body.get('input')
    body['output'] = translation.translate(action, input)
    return jsonify(body)


if __name__ == '__main__':
    app = create_app()
    app.run()
