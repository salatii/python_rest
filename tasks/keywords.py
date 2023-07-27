from transformers import T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT

def init():
    print('init keyword model')
    global model, tokenizer
    model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords")
    tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords")


def init2():
    print('init keyBERT')
    global kw_model
    kw_model = KeyBERT()

def extract(input):
    task_prefix = "Keywords: "
    input_sequences = [task_prefix + input]
    input_ids = tokenizer(
        input_sequences, return_tensors="pt", truncation=True
    ).input_ids
    output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
    predicted = tokenizer.decode(output[0], skip_special_tokens=True)
    print("--->", predicted)
    return predicted

def extract2(input):
    keywords = kw_model.extract_keywords(input)
    return keywords

