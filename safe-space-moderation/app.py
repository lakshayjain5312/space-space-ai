import numpy as np
from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import re
from nltk.stem import WordNetLemmatizer
import nltk
import emoji
from contractions import fix as contractions_fix
from nltk.tokenize import RegexpTokenizer
from googleapiclient import discovery
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz, process as rapidfuzz_process
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from detoxify import Detoxify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# Initialize the model and tokenizer
# Load model directly








# Load tokenizer and model

# 2. Load the fast tokenizer and model

detoxify_model=Detoxify('unbiased')



model_name_of_hindi = "hindi-toxicity-model"
tokenizer_of_hindi = AutoTokenizer.from_pretrained(model_name_of_hindi, use_fast=True)
model_of_hindi = AutoModelForSequenceClassification.from_pretrained(model_name_of_hindi)
model_of_hindi.eval() 



# Hindi-to-English translation model


# Add this before creating the Flask app
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


API_KEY = 'AIzaSyC_Eatd353BWOJkeXboVg1Q8aElI5nJEz4'

# Initialize Perspective API client
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,)




app = Flask(__name__)


def hindi_toxicity_check(text):
        # Tokenize the input text
        inputs = tokenizer_of_hindi(text, return_tensors="pt", max_length=256,truncation=True, padding=True)
        # Move inputs to the selected device

        
        # Disable gradients for faster inference
        with torch.no_grad():
            outputs = model_of_hindi(**inputs)
        
        # Convert logits to probabilities
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Map the predicted class index to a label (if available)
        if hasattr(model_of_hindi.config, "id2label"):
            label = model_of_hindi.config.id2label[predicted_class]
        else:
            label = str(predicted_class)
        
        return label



def transliterate_to_hindi(text):
    hindi_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    return hindi_text

def preprocess_text(text):
    # Ordered replacement patterns (specific to general)
    l33t_replacements = [
        # URLs, emojis, and encoding handled first in normalization
        
        # ==== Multi-character patterns ====
        (r"¯\\_\(ツ\)_/¯", ""),  # Remove shrug emoji
        
        # ==== Special character replacements ====
        (r"@", "a"), (r"!", "i"), (r"\$", "s"),
        (r"\|_\|", "u"), (r"\/", "v"), (r"><", "x"),
        (r"\[\]", "o"), (r"\(\)", "o"), (r"\*", "o"),
        (r"ß", "b"),
        
        # ==== Contextual digit-letter hybrids (prevent duplicates) ====
        (r"3e", "e"),  # Fixes "chutiy3e" → "chutiye"
        (r"4a", "a"), (r"1i", "i"), (r"0o", "o"),
        (r"7t", "t"), (r"5s", "s"), (r"2z", "z"),
        
        # ==== Full l33t word replacements ====
        (r"\bd00d\b", "dude"), (r"\bj00\b", "you"),
        (r"\bz3\b", "the"), (r"\bh4x\b", "hack"),
        (r"\bn00b\b", "noob"), (r"\br00t\b", "root"),
        (r"\bstr8\b", "straight"), (r"\bl8r\b", "later"),
        (r"\bh4xx0r\b", "hacker"), (r"\bl33t\b", "leet"),
        
        # ==== Digit replacements ====
        (r"0", "o"), (r"1", "i"), (r"3", "e"),
        (r"4", "a"), (r"5", "s"), (r"7", "t"),
        (r"2", "z"),
    ]

    def _normalize(text):
        # Handle encoding/emojis first
        text = text.encode('utf-8', 'replace').decode('utf-8')
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)  # Basic emoticons
        
        # Apply l33t replacements
        for pattern, replacement in l33t_replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Standardize apostrophes
        text = re.sub(r"’", "'", text)  
        
        # Phonetic/slang substitutions (reduced substitutions)
        phonetic_map = {
            r"\bthx\b": "thanks",
            r"\btbh\b": "to be honest", 
            r"\bbrb\b": "be right back",
        }
        for pattern, replacement in phonetic_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text

    # ==== Main Pipeline ====
    # Initial cleanup
    text = text.lower().strip()
    
    # Normalization layer
    text = _normalize(text)
    
    # Remove URLs but keep mentions and hashtags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Allow hyphens and underscores in words
    text = re.sub(r"[^\w\s'-_]", '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenization (keep original words without lemmatization)
    tokenizer = RegexpTokenizer(r"\w+[\w'-]*\w+|\w")  # Expanded to include hyphens
    words = tokenizer.tokenize(text)
    
    # Mild repetition reduction (only for 4+ repeats)
    def _reduce_repetitions(text):
        # Reduce 4+ repeats to 2 chars
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        return text
    
    text = ' '.join(words)
    text = _reduce_repetitions(text)
    
    # Preserve exceptions without changes
    exceptions = {
        r'\bmoon\b': 'moon', 
        r'\bcoffee\b': 'coffee',
        r'\bsweet\b': 'sweet'
    }
    for pattern, replacement in exceptions.items():
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def contains_offensive_language(text):
    # Preprocess the text
    

    # Tokenize the text
    tokens = text.split()

    # Load offensive words list
    with open('offensive_words.txt', 'r') as f:
        offensive_words = [line.strip() for line in f]

    # Initialize stemmer
    stemmer = PorterStemmer()
    offensive_stems = [stemmer.stem(word) for word in offensive_words]

    # Stem the tokens
    token_stems = [stemmer.stem(token) for token in tokens]
    # Check for offensive words using exact and fuzzy matching
    for token in token_stems:
        # Exact match
        if token in offensive_stems:
            return True
        # Fuzzy match
        matches = rapidfuzz_process.extract(token, offensive_stems, scorer=fuzz.ratio, limit=1)
        
        if matches and matches[0][1] > 80: 
            print(matches)
            print(matches[0][1]) 
            return True
    return False


def process_msg(msg):
    if msg == "hi":
        response = "Hello, Welcome to the cyberbullying detection bot!"
    else:
        
        msg = preprocess_text(msg)
        print(msg)
        if contains_offensive_language(msg):
            print('edge')
            return True
        if detoxify_model.predict(msg)['toxicity'] > 0.75:
            print('eng')
            return True
        transliterated_hindi_msg=transliterate_to_hindi(msg)
        if hindi_toxicity_check(transliterated_hindi_msg) == 'LABEL_1':
            print('hindi')
            return True
        # msg = [msg]
        # # List of stopwords 
        # my_file = open("stopwords.txt", "r")
        # content = my_file.read()
        # content_list = content.split("\n")
        # my_file.close()

        # tfidf_vector = TfidfVectorizer(stop_words = content_list, lowercase = True,vocabulary=pickle.load(open("tfidf_vector_vocabulary.pkl", "rb")))
        # data=tfidf_vector.fit_transform(msg)
        # model = pickle.load(open("LinearSVC.pkl", 'rb'))
        # print(data)
        # pred = model.predict(data)
        # response = pred[0]
        # print(response)
        # if response == 1:
        #     return False
        

    return False 
   


def check_with_perspective_api(text):
    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['en', 'hi'] 
    }
    response = client.comments().analyze(body=analyze_request).execute()
    score = response['attributeScores']['TOXICITY']['summaryScore']['value']
    return score

@app.route("/api/v1/toxicity", methods=["POST"])
def testing():
    data = request.json
    msg = data.get('msg', '')
    
    print('custom api is now running.....')
    custom_ml_response = process_msg(msg)
    if custom_ml_response:
        result={'toxicity': 1,'msg':msg}
    else:
        result={'toxicity': 0}
    
    # print('perspective is now running.....')
    
    # perspective_score = check_with_perspective_api(msg)

    # if perspective_score > 0.7:
    #     result={'toxicity': 1,'msg':msg}
    
    
    return jsonify(result)

    
if __name__ == "__main__":
    app.run(debug=True)


