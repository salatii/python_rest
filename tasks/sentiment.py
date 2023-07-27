from transformers import pipeline



def init():
    print('init sentiment model')
    global classifier
    classifier = pipeline("sentiment-analysis")


def sentiment(input):
    return classifier(input)