from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
import math
import torch
import nltk
import os

#nltk.download('punkt')
os.environ['CURL_CA_BUNDLE'] = ''

translation_en_de = None
tokenizer = None
model = None
device = None


def init():
    print('init model')
    global translation_en_de, tokenizer, model, device
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

   # translation_en_de = pipeline("translation_en_to_de", model_max_length=512)

    mname = 'Helsinki-NLP/opus-mt-de-en'
    tokenizer = MarianTokenizer.from_pretrained(mname, model_max_length=512)
    model = MarianMTModel.from_pretrained(mname)
    model.to(device)


def translate(action, input_text):
    print('start translation')
    if (tokenizer is not None) and (model is not None) and (device is not None):
        if input_text:
            if action == 'translate_de_en':
                lt = LineTokenizer()
                batch_size = 8
                paragraphs = lt.tokenize(input_text)
                translated_paragraphs = []

                for paragraph in paragraphs:
                    sentences = sent_tokenize(paragraph)
                    batches = math.ceil(len(sentences) / batch_size)
                    translated = []
                    for i in range(batches):
                        sent_batch = sentences[i * batch_size:(i + 1) * batch_size]
                        model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True).to(
                            device)
                        with torch.no_grad():
                            translated_batch = model.generate(**model_inputs)
                        translated += translated_batch
                    translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                    translated_paragraphs += [" ".join(translated)]
                translated_text = "\n".join(translated_paragraphs)
        return translated_text
    else:
        print('problem')


init()
print(translate("translate_de_en", "Das ist ein Beispiel Text"))

