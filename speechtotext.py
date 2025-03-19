

import speech_recognition as sr
import os 
import language_tool_python
from difflib import SequenceMatcher
from pydub import AudioSegment
from pydub.utils import which
import librosa
import soundfile as sf
# AudioSegment.converter = which("ffmpeg")
# def audio_to_text(audio_path):
#     """Convert audio file to text."""
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio_data = recognizer.record(source)
#     return recognizer.recognize_google(audio_data)


def convert_to_wav(audio_path):
    """Converts an audio file to WAV format if needed."""
    if not audio_path.lower().endswith('.wav'):
        new_audio_path = audio_path.rsplit('.', 1)[0] + ".wav"
        audio = AudioSegment.from_file(audio_path)
        audio.export(new_audio_path, format="wav")
        return new_audio_path
    return audio_path

# def convert_to_wav(audio_path):
#     """Converts an audio file to WAV format if needed."""
#     if not audio_path.lower().endswith('.wav'):
#         new_audio_path = audio_path.rsplit('.', 1)[0] + ".wav"
#         y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
#         sf.write(new_audio_path, y, sr)  # Save as WAV
#         return new_audio_path
#     return audio_path



def audio_to_text(audio_path):
    """Convert audio file to text."""
    audio_path = convert_to_wav(audio_path)  # Convert to WAV if necessary
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    return recognizer.recognize_google(audio_data)


drive_to_local_mapping = {
    "https://drive.google.com/file/d/1dHHceWxl9a-KRsxqRktE5_pIUeDNOz4b/view?usp=sharing": r"audiofiles\1001_DFA_NEU_XX.wav",
    "https://drive.google.com/file/d/1dHHceWxl9a-KRsxqRktE5_pIUeDNOz4b/view?usp=sharing": r"audiofiles\harvard.wav",
    "https://drive.google.com/file/d/1QjOoKz1_u15KO-yXyVV5GWIxkpTqkgAU/view?usp=drive_link": r"audiofiles\Recording.wav"
}

def get_local_path(drive_url):
    """Fetches the local file path corresponding to a given Google Drive URL."""
    return drive_to_local_mapping.get(drive_url, None)

# def process_file(drive_url):
#     """Processes the file by extracting text from it."""
#     local_path = get_local_path(drive_url)
#     if local_path and os.path.exists(local_path):
#         print(f"Processing file at: {local_path}")
#         audiototext = audio_to_text(local_path)
#         return audiototext
#     else:
#         print("File not found or mapping doesn't exist!")
#         return None


def process_file(drive_url):
    """Processes the file by extracting text from it."""
    if "drive.google.com" in drive_url:
        local_path = get_local_path(drive_url)
        if local_path and os.path.exists(local_path):
            print(f"Processing file at: {local_path}")
            return audio_to_text(local_path)
        else:
            print("File not found or mapping doesn't exist!")
            return None
    else:
        print(f"Processing file directly from URL: {drive_url}")
        return audio_to_text(drive_url)


# Example Usage
# drive_url = "https://drive.google.com/file/d/1dHHceWxl9a-KRsxqRktE5_pIUeDNOz4b/view?usp=sharing"
# audio_input_text = process_file(drive_url)
# print (audio_input_text)


letter_reversals = {
    'b': 'd', 'd': 'b',
    'p': 'q', 'q': 'p',
    'm': 'w', 'w': 'm',
    'n': 'u', 'u': 'n'
}

def compare_texts(input_text, extracted_text):
    letter_reversal_feedback = set()
    spelling_mistake_feedback = set()
    input_words = input_text.split()
    extracted_words = extracted_text.split()
    
    # Identify letter reversals and spelling mistakes
    for input_word, extracted_word in zip(input_words, extracted_words):
        if input_word.lower() != extracted_word.lower():
            matcher = SequenceMatcher(None, input_word, extracted_word)
            differences = matcher.get_opcodes()
            for tag, i1, i2, j1, j2 in differences:
                if tag == "replace":
                    # Check for letter reversals in the replaced word
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        if input_word[i].lower() in letter_reversals:
                            reversed_letter = letter_reversals[input_word[i].lower()]
                            if extracted_word[j].lower() == reversed_letter:
                                letter_reversal_feedback.add(
                                    f"'{input_word[i]}' was written as '{reversed_letter}' in word '{input_word}'."
                                )
                    # If not a letter reversal, add it as a spelling mistake
                    else:
                        spelling_mistake_feedback.add(
                            f"Expected '{input_word}', but wrote '{extracted_word}'."
                        )

    # Sorting: Letter reversals first, then spelling mistakes
    return "\n".join(sorted(letter_reversal_feedback) + sorted(spelling_mistake_feedback))


# Example usage
# input_text1 = 'The quick brown fox jumped over the lazy dog'  #audio input text for comparison
# extracted_text1 = 'Te puik drwou fxo jumepd oever the layz bog' #ocr extracted text
# feedback1 = compare_texts(input_text1, extracted_text1)

# print("\nFeedback:\n")
# print(feedback1)
