from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
import math
import torch
import nltk
import os

#nltk.download('punkt')
os.environ['CURL_CA_BUNDLE'] = ''

def init():
    print('init model')
    global device, tokenizer_EN_DE, model_EN_DE, tokenizer_DE_EN, model_DE_EN
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # Helsinki-NLP/opus-mt-de-en
    model_path = "../models/DE_EN"
    tokenizer_DE_EN = MarianTokenizer.from_pretrained(model_path, model_max_length=512)
    model_DE_EN = MarianMTModel.from_pretrained(model_path)
    model_DE_EN.to(device)

    # Helsinki-NLP/opus-mt-en-de
    model_path = "../models/EN_DE"
    tokenizer_EN_DE = MarianTokenizer.from_pretrained(model_path, model_max_length=512)
    model_EN_DE = MarianMTModel.from_pretrained(model_path)
    model_EN_DE.to(device)


def translate(action, input_text):
    print('start translation')
    if action == 'translate_de_en':
        if (tokenizer_DE_EN is not None) and (model_DE_EN is not None) and (device is not None):
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
                    model_inputs = tokenizer_DE_EN(sent_batch, return_tensors="pt", padding=True, truncation=True).to(
                        device)
                    with torch.no_grad():
                        translated_batch = model_DE_EN.generate(**model_inputs)
                    translated += translated_batch
                translated = [tokenizer_DE_EN.decode(t, skip_special_tokens=True) for t in translated]
                translated_paragraphs += [" ".join(translated)]
            translated_text = "\n".join(translated_paragraphs)
        else:
            print('german to english model error')
    elif action == 'translate_en_de':
        #TODO repect also word length
        if (tokenizer_EN_DE is not None) and (model_EN_DE is not None) and (device is not None):
            translated = model_EN_DE.generate(
                **tokenizer_EN_DE(input_text, return_tensors="pt", padding=True))
            translated_text = ''
            for t in translated:
                translated_text += (tokenizer_EN_DE.decode(t, skip_special_tokens=True))
        else:
            print('english to german model error')

    return translated_text


