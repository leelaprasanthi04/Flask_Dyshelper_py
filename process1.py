# Standard library imports
import os
import time
import random

# Third-party imports
import streamlit as st
from PIL import Image
import pandas as pd
import requests
from textblob import TextBlob
import language_tool_python
import pyttsx3
import eng_to_ipa as ipa
import re
#from phonetisch import Soundex
#from soundex import Soundex
from py4Soundex.code import Soundex
from metaphone import doublemetaphone as Metaphone
from phonetisch import caverphone
from textblob import TextBlob
from fuzzywuzzy import fuzz
from nysiis import NYSIIS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib
import math
import collections
from collections import Counter
from google.cloud import vision


my_tool = language_tool_python.LanguageTool('en-US')

# Functions for feature array extraction

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def caverphone(word):
    """ Caverphone algorithm to generate a phonetic representation of a word. """
    word = word.lower()

    # Step 1: Handle basic patterns
    word = re.sub(r'ough', 'uf', word)
    word = re.sub(r'ph', 'f', word)
    word = re.sub(r'([aeiou])h', r'\1', word)
    word = re.sub(r'([aeiou])w', r'\1', word)
    word = re.sub(r'([aeiou])y', r'\1', word)

    # Step 2: Apply sound pattern rules
    word = re.sub(r'[aeiou]', '', word)  # Remove vowels

    # Handle specific cases in the Caverphone rules
    word = re.sub(r'ck', 'k', word)
    word = re.sub(r'[^aeiou]y', '', word)

    # Handle ending cases
    word = re.sub(r'([aeiou])$', '', word)

    return word

def spelling_accuracy(extracted_text):
   # print('hey spelling accuracy')
    spell_corrected = TextBlob(extracted_text).correct()
    return ((len(extracted_text) - (levenshtein(extracted_text, spell_corrected)))/(len(extracted_text)+1))*100

def gramatical_accuracy(extracted_text):
    #print('hey grammatical accuracy')
    spell_corrected = TextBlob(extracted_text).correct()
    correct_text = my_tool.correct(spell_corrected)
    extracted_text_set = set(spell_corrected.split(" "))
    correct_text_set = set(correct_text.split(" "))
    n = max(len(extracted_text_set - correct_text_set),
            len(correct_text_set - extracted_text_set))
    return ((len(spell_corrected) - n)/(len(spell_corrected)+1))*100

def percentage_of_corrections_textblob(extracted_text):
   # print('percentage_of_corrections_textblob')
    words = extracted_text.split()
    corrected_words = TextBlob(extracted_text).correct().split()
    
    incorrect_count = sum(1 for w1, w2 in zip(words, corrected_words) if w1 != w2)
    
    return (incorrect_count / len(words)) * 100 if words else 0

def percentage_of_phonetic_accuraccy(extracted_text: str):
   # print('percentage_of_phonetic accuracy')
    soundex = Soundex
    metaphone = Metaphone
   # caverphone = caverphone
    nysiis = NYSIIS()
    spell_corrected = TextBlob(extracted_text).correct()

    extracted_text_list = extracted_text.split(" ")
   # extracted_phonetics_soundex = [soundex.encode(string) for string in extracted_text_list]
   # extracted_phonetics_metaphone = [metaphone.encode(string) for string in extracted_text_list]
   # extracted_phonetics_caverphone = [caverphone.encode( string) for string in extracted_text_list]
    #extracted_phonetics_nysiis = [nysiis.encode(string) for string in extracted_text_list]
    extracted_phonetics_soundex = [soundex(string)[0] for string in extracted_text_list] 
    extracted_phonetics_metaphone = [metaphone(string)[0] for string in extracted_text_list]  # ✅ Corrected
    extracted_phonetics_caverphone = [caverphone(string) for string in extracted_text_list]
    extracted_phonetics_nysiis = [nysiis.encode(string) for string in extracted_text_list]

    extracted_soundex_string = " ".join(extracted_phonetics_soundex)
    extracted_metaphone_string = " ".join(extracted_phonetics_metaphone)
    extracted_caverphone_string = " ".join(extracted_phonetics_caverphone)
    extracted_nysiis_string = " ".join(extracted_phonetics_nysiis)

    spell_corrected_list = spell_corrected.split(" ")

    #print(spell_corrected_list)

    spell_corrected_phonetics_soundex = [soundex(string)[0] for string in spell_corrected_list] 
    spell_corrected_phonetics_metaphone = [metaphone(string)[0] for string in spell_corrected_list]  # ✅ Corrected
    spell_corrected_phonetics_caverphone = [caverphone(string) for string in spell_corrected_list]
    spell_corrected_phonetics_nysiis = [nysiis.encode(string) for string in spell_corrected_list]

    #spell_corrected_phonetics_soundex = [soundex.encode(string) for string in spell_corrected_list]
    #spell_corrected_phonetics_metaphone = [metaphone.encode(string) for string in spell_corrected_list]
    #spell_corrected_phonetics_caverphone = [caverphone.encode(string) for string in spell_corrected_list]
    #spell_corrected_phonetics_nysiis = [nysiis.encode(string) for string in spell_corrected_list]
    #print(spell_corrected_phonetics_metaphone)
    spell_corrected_soundex_string = " ".join(spell_corrected_phonetics_soundex)
    spell_corrected_metaphone_string = " ".join(spell_corrected_phonetics_metaphone)
    spell_corrected_caverphone_string = " ".join(spell_corrected_phonetics_caverphone)
    spell_corrected_nysiis_string = " ".join(spell_corrected_phonetics_nysiis)

    soundex_score = (len(extracted_soundex_string)-(levenshtein(extracted_soundex_string,spell_corrected_soundex_string)))/(len(extracted_soundex_string)+1)
  
    metaphone_score = (len(extracted_metaphone_string)-(levenshtein(extracted_metaphone_string, spell_corrected_metaphone_string)))/(len(extracted_metaphone_string)+1)
 
    caverphone_score = (len(extracted_caverphone_string)-(levenshtein(extracted_caverphone_string,spell_corrected_caverphone_string)))/(len(extracted_caverphone_string)+1)

    nysiis_score = (len(extracted_nysiis_string)-(levenshtein(extracted_nysiis_string, spell_corrected_nysiis_string)))/(len(extracted_nysiis_string)+1)
   
    return ((0.5*caverphone_score + 0.2*soundex_score + 0.2*metaphone_score + 0.1 * nysiis_score))*100

def get_feature_array(extracted_text):
    feature_array = []
    
    feature_array.append(spelling_accuracy(extracted_text))
    feature_array.append(gramatical_accuracy(extracted_text))
    feature_array.append(percentage_of_corrections_textblob(extracted_text))
    feature_array.append(percentage_of_phonetic_accuraccy(extracted_text))
    
    return feature_array

#function for predicting the presence of disease
def dys_pred(feature_array) :
    loaded_data = {'models': {'logistic': LogisticRegression(random_state=42),
                           'rf': RandomForestClassifier(random_state=42), 
                           'gb': GradientBoostingClassifier(random_state=42), 
                           'svm': SVC(kernel='linear', probability=True, random_state=42)}, 
                           'best_weights': np.array([0.34602922, 0.00712688, 0.0066998 , 0.6401441 ])}


    model_logistic = joblib.load("models/model_logistic.joblib")
    model_rf = joblib.load("models/model_rf.joblib")
    model_gb = joblib.load("models/model_gb.joblib")
    model_svm = joblib.load("models/model_svm.joblib")


    models = {
    "logistic": model_logistic,
    "rf": model_rf,
    "gb": model_gb,
    "svm": model_svm
    }
    feature_array = np.array(feature_array)
    #feature_array = get_feature_array(query)
    # feature_array = np.array([78.57, 92.30, 50.0, 81.502])
    best_weights = np.array([0.34602922, 0.00712688, 0.0066998,   0.6401441])

    # Define feature names (Replace with actual feature names)
    feature_names = ["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections", "percentage_of_phonetic_accuraccy"]

    # Assume X_new is the new test sample(s)
    X_new = feature_array.reshape(1, -1) 
    X_new = np.array([78.57, 92.30, 50.0, 81.502]).reshape(1, -1)  # Convert to 2D array

    # Convert X_new into a DataFrame with feature names
    X_new = pd.DataFrame(X_new, columns=feature_names)
    # print("Input DataFrame:\n", X_new)

    # Get probability predictions from each model
    logistic_probs = models["logistic"].predict_proba(X_new)[:, 1]
    rf_probs = models["rf"].predict_proba(X_new)[:, 1]
    gb_probs = models["gb"].predict_proba(X_new)[:, 1]
    svm_probs = models["svm"].predict_proba(X_new)[:, 1]

    # Combine probabilities into a single matrix
    base_probs = np.vstack((logistic_probs, rf_probs, gb_probs, svm_probs)).T

    # Compute weighted probabilities
    ensemble_probs = np.dot(base_probs, best_weights)

    # Convert probabilities to class predictions (Threshold = 0.5)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # Display the result
    print("Ensemble Predictions:", ensemble_preds)
    if ensemble_preds[0] == 1:
       # print("Prediction: Dyslexia detected")
       return 'Dyslexia Presence Detected'
    else:
       # print("Prediction: No Dyslexia detected")
       return 'Dyslexia Presence Not Detected'

    return ensemble_preds

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'ocr.json')
WORD = re.compile(r"\w+")


def detect_text1(image_path):
  
    client = vision.ImageAnnotatorClient()

    #content = image_file.read()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Use document_text_detection for dense text (you can switch to text_detection for non-dense text)
    response = client.document_text_detection(image=image)

    # Check for errors in the response
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # Get the full detected text
    full_text = response.full_text_annotation.text

    return full_text  # Return the detected text



