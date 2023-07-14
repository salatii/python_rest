<<<<<<< HEAD
from transformers import MarianMTModel, MarianTokenizer
import os
os.environ['CURL_CA_BUNDLE'] = ''

text_DE = "Hallo kannst du mir diesen Text übersetzen. Ich bin leider nicht so gut in Englisch."

#Helsinki-NLP/opus-mt-de-en
model_path = ".\models\DE_EN"
tokenizer_DE_EN = MarianTokenizer.from_pretrained(model_path)
model_DE_EN = MarianMTModel.from_pretrained(model_path)

translated = model_DE_EN.generate(**tokenizer_DE_EN(text_DE, return_tensors="pt", padding=True))

#model.save_pretrained(".\models\DE_EN")
#tokenizer_DE_EN.save_pretrained(".\models\DE_EN")

translated_str = ''
for t in translated:
    translated_str += (tokenizer_DE_EN.decode(t, skip_special_tokens=True))

print(translated_str)
#-------------------------------------------------------------------------------------------------------------
#Helsinki-NLP/opus-mt-en-de
model_path = ".\models\EN_DE"
tokenizer_EN_DE = MarianTokenizer.from_pretrained(model_path)
model_EN_DE = MarianMTModel.from_pretrained(model_path)

translated = model_EN_DE.generate(**tokenizer_EN_DE(translated_str, return_tensors="pt", padding=True))

#model_EN_DE.save_pretrained(".\models\EN_DE")
#tokenizer_EN_DE.save_pretrained(".\models\EN_DE")

translated_back = ''
for t in translated:
    translated_back += (tokenizer_EN_DE.decode(t, skip_special_tokens=True))

=======
from transformers import MarianMTModel, MarianTokenizer
import os
os.environ['CURL_CA_BUNDLE'] = ''

text_DE = "Hallo kannst du mir diesen Text übersetzen. Ich bin leider nicht so gut in Englisch."

#Helsinki-NLP/opus-mt-de-en
model_path = ".\models\DE_EN"
tokenizer_DE_EN = MarianTokenizer.from_pretrained(model_path)
model_DE_EN = MarianMTModel.from_pretrained(model_path)

translated = model_DE_EN.generate(**tokenizer_DE_EN(text_DE, return_tensors="pt", padding=True))

#model.save_pretrained(".\models\DE_EN")
#tokenizer_DE_EN.save_pretrained(".\models\DE_EN")

translated_str = ''
for t in translated:
    translated_str += (tokenizer_DE_EN.decode(t, skip_special_tokens=True))

print(translated_str)
#-------------------------------------------------------------------------------------------------------------
#Helsinki-NLP/opus-mt-en-de
model_path = ".\models\EN_DE"
tokenizer_EN_DE = MarianTokenizer.from_pretrained(model_path)
model_EN_DE = MarianMTModel.from_pretrained(model_path)

translated = model_EN_DE.generate(**tokenizer_EN_DE(translated_str, return_tensors="pt", padding=True))

#model_EN_DE.save_pretrained(".\models\EN_DE")
#tokenizer_EN_DE.save_pretrained(".\models\EN_DE")

translated_back = ''
for t in translated:
    translated_back += (tokenizer_EN_DE.decode(t, skip_special_tokens=True))

>>>>>>> 20fda42b756a8e6715294efb3c725f4edc168d34
print(translated_back)