import langid

def detect(input):
    detected_language, confidence = langid.classify(input)
    return detected_language, confidence