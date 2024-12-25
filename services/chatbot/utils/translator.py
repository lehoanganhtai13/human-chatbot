import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

from deep_translator import GoogleTranslator
from langdetect import detect, detect_langs
import time

LANGUAGE_CODES = {
    "afrikaans": "af", "albanian": "sq", "amharic": "am", "arabic": "ar", 
    "armenian": "hy", "assamese": "as", "aymara": "ay", "azerbaijani": "az", 
    "bambara": "bm", "basque": "eu", "belarusian": "be", "bengali": "bn", 
    "bhojpuri": "bh", "bosnian": "bs", "bulgarian": "bg", "catalan": "ca", 
    "cebuano": "ceb", "chichewa": "ny", "chinese (simplified)": "zh-cn", 
    "chinese (traditional)": "zh-tw", "corsican": "co", "croatian": "hr", 
    "czech": "cs", "danish": "da", "dhivehi": "dv", "dogri": "doi", 
    "dutch": "nl", "english": "en", "esperanto": "eo", "estonian": "et", 
    "ewe": "ee", "filipino": "fil", "finnish": "fi", "french": "fr", 
    "frisian": "fy", "galician": "gl", "georgian": "ka", "german": "de", 
    "greek": "el", "guarani": "gn", "gujarati": "gu", "haitian creole": "ht", 
    "hausa": "ha", "hawaiian": "haw", "hebrew": "he", "hindi": "hi", 
    "hmong": "hmn", "hungarian": "hu", "icelandic": "is", "igbo": "ig", 
    "ilocano": "ilo", "indonesian": "id", "irish": "ga", "italian": "it", 
    "japanese": "ja", "javanese": "jv", "kannada": "kn", "kazakh": "kk", 
    "khmer": "km", "kinyarwanda": "rw", "konkani": "kok", "korean": "ko", 
    "krio": "kri", "kurdish (kurmanji)": "ku", "kurdish (sorani)": "ckb", 
    "kyrgyz": "ky", "lao": "lo", "latin": "la", "latvian": "lv", 
    "lingala": "ln", "lithuanian": "lt", "luganda": "lg", "luxembourgish": "lb", 
    "macedonian": "mk", "maithili": "mai", "malagasy": "mg", "malay": "ms", 
    "malayalam": "ml", "maltese": "mt", "maori": "mi", "marathi": "mr", 
    "meiteilon (manipuri)": "mni", "mizo": "lus", "mongolian": "mn", 
    "myanmar": "my", "nepali": "ne", "norwegian": "no", "odia (oriya)": "or", 
    "oromo": "om", "pashto": "ps", "persian": "fa", "polish": "pl", 
    "portuguese": "pt", "punjabi": "pa", "quechua": "qu", "romanian": "ro", 
    "russian": "ru", "samoan": "sm", "sanskrit": "sa", "scots gaelic": "gd", 
    "sepedi": "nso", "serbian": "sr", "sesotho": "st", "shona": "sn", 
    "sindhi": "sd", "sinhala": "si", "slovak": "sk", "slovenian": "sl", 
    "somali": "so", "spanish": "es", "sundanese": "su", "swahili": "sw", 
    "swedish": "sv", "tajik": "tg", "tamil": "ta", "tatar": "tt", 
    "telugu": "te", "thai": "th", "tigrinya": "ti", "tsonga": "ts", 
    "turkish": "tr", "turkmen": "tk", "twi": "tw", "ukrainian": "uk", 
    "urdu": "ur", "uyghur": "ug", "uzbek": "uz", "vietnamese": "vi", 
    "welsh": "cy", "xhosa": "xh", "yiddish": "yi", "yoruba": "yo", "zulu": "zu"
}

class Translator:
    """A class to translate text and detect the language of the text."""
    def __init__(self, source: str = "auto", target: str = "english", capitalize_sentences: bool = False):
        self.translator = GoogleTranslator(source=source, target=LANGUAGE_CODES[target.lower()])
        self._capitalize_sentences = capitalize_sentences
        self.target = target.lower()

    @staticmethod
    def get_supported_languages():
        """Get the supported languages for translation."""

        translator = GoogleTranslator(source="auto", target="en")
        return translator.get_supported_languages()

    def capitalize_sentences(self, text):
        """Capitalize the first letter of each sentence in the text."""

        sentences = sent_tokenize(text)
        capitalized_sentences = [sentence.capitalize() for sentence in sentences]
        return " ".join(capitalized_sentences)
    
    def detect_language(self, text):
        """Detect the language of the text."""

        language = detect(text)
        detection_score = detect_langs(text)[0].prob

        # Return the name of the language
        for lang_name, lang_code in LANGUAGE_CODES.items():
            if lang_code == language:
                language = lang_name
                break

        return language, detection_score
    
    def translate(self, text: str, force_target: bool = False):
        """Translate the text into the target language."""

        translated_text = ""
        if force_target:
            translated_text = self.translator.translate(text)
        else:
            # If the text is already in the target language, return it as is
            language, _ = self.detect_language(text)
            if language == self.target:
                return text

        # Capitalize the translated sentences if required
        if self._capitalize_sentences:
            translated_text = self.capitalize_sentences(translated_text)

        return translated_text