from flask import Flask, request, jsonify
import feedparser
import requests
import os
from tasks import translation
from tasks import summarization
from tasks import sentiment
from tasks import lng_detection
from tasks import keywords

openaikey = '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'

#nltk.download('punkt')
os.environ['CURL_CA_BUNDLE'] = ''

requests.Session().verify = False

# TODO execute before first request and load models
def init_models():
    translation.init()
    summarization.init()
    sentiment.init()
    keywords.init()
    keywords.init2()


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
    body['output'] = keywords.extract2(input)
    return jsonify(body)


def get_summary(body):
    input = body.get('input')
    body['output'] = summarization.summarize(input)
    return jsonify(body)


def get_sentiment(body):
    input = body.get('input')
    body['output'] = sentiment.sentiment(input)
    return jsonify(body)


def get_language(body):
    input = body.get('input')
    body['output'] = lng_detection.detect(input)
    return jsonify(body)


def get_translation(body):
    action = body['action']
    input = body.get('input')
    body['output'] = translation.translate(action, input)
    return jsonify(body)


if __name__ == '__main__':
    app = create_app()
    app.run()
