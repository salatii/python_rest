from flask import Flask, request
import feedparser
from keybert import KeyBERT
from transformers import pipeline
import requests

requests.Session().verify = False

app = Flask(__name__)

@app.route("/")
def hello_world():
    #TODO welche uuids muss man tätigen
    return "Hello, World!"

@app.route('/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    content_type = request.headers.get('Content-Type')
    content_check = check_content_type(content_type)
    json = None
    if content_check:
        json = request.json
    print(uuid)
    print(json)
    print(json is not None)

    if uuid == 'rss':
        return get_rss()
    elif uuid == 'sentiment':
        if json is not None:
            return get_sentiment(json['content'])
        else:
            return 'error'
    elif uuid == 'summarize':
        if json is not None:
            return get_summary(json['content'])
        else:
            return 'error'
    elif uuid == 'keywords':
        if json is not None:
            return get_keywords(json['content'])
        else:
            return 'error'
    elif uuid == 'translate':
        if json is not None:
            return 'translate'
        else:
            return 'error'
    else:
        return 'error'

def check_content_type(content_type):
    if (content_type == 'application/json'):
        return True
    else:
        return False

def get_rss():
    rss_url = 'https://www.heise.de/rss/heise.rdf'
    feed = feedparser.parse(rss_url).entries
    #return json.dumps(feed)
    return feed

def get_keywords(content):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(content)
    print(keywords)
    return keywords

def get_summary(content):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(content, max_length=130, min_length=30, do_sample=False)

def get_sentiment(content):
    print("Hallo")
    classifier = pipeline("sentiment-analysis")
    print("Hallo1")
    return classifier(content)

