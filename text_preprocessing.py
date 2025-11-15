import re
import spacy
from pymorphy2 import MorphAnalyzer
from config import PREPROCESSING_CONFIG

class TextPreprocessor:
    def __init__(self, tokenization_method="whitespace"):
        self.tokenization_method = tokenization_method
        self.morph = MorphAnalyzer()
        self.bpe_processor = None
        
        # Загрузка модели для лемматизации
        try:
            self.nlp = spacy.load("ru_core_news_sm")
        except OSError:
            self.nlp = None
            print("spacy model not found, using pymorphy2 only")
    
    def load_bpe_model(self, model_path):
        """Загрузка обученной BPE модели"""
        try:
            # Попробуем использовать subword-nmt если установлен
            from subword_nmt import apply_bpe
            with open(model_path, 'r', encoding='utf-8') as f:
                self.bpe_processor = apply_bpe.BPE(f)
        except ImportError:
            print("subword-nmt not available, using simple BPE implementation")
            self.bpe_processor = SimpleBPEProcessor()
            self.bpe_processor.load_model(model_path)
    
    def clean_text(self, text):
        """Очистка текста"""
        if not isinstance(text, str):
            return ""
        # Удаление специальных символов, цифр, лишних пробелов
        text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def lemmatize(self, tokens):
        """Лемматизация токенов"""
        lemmas = []
        for token in tokens:
            if len(token) < PREPROCESSING_CONFIG["min_token_length"]:
                continue
                
            if self.nlp:
                # Используем spacy если доступно
                try:
                    doc = self.nlp(token)
                    if len(doc) > 0:
                        lemmas.append(doc[0].lemma_)
                except:
                    # Fallback to pymorphy2
                    parsed = self.morph.parse(token)[0]
                    lemmas.append(parsed.normal_form)
            else:
                # Используем pymorphy2
                parsed = self.morph.parse(token)[0]
                lemmas.append(parsed.normal_form)
        return lemmas
    
    def tokenize_whitespace(self, text):
        """Токенизация по пробелам"""
        return text.split()
    
    def tokenize_regex(self, text):
        """Токенизация с помощью регулярных выражений"""
        tokens = re.findall(r'\b[а-яёa-z]{2,}\b', text.lower())
        return tokens
    
    def tokenize_bpe(self, text):
        """Токенизация с использованием BPE"""
        if self.bpe_processor:
            if hasattr(self.bpe_processor, 'segment'):
                # Для subword-nmt
                return self.bpe_processor.segment(text).split()
            else:
                # Для нашей простой реализации
                return self.bpe_processor.encode(text)
        else:
            return self.tokenize_regex(text)
    
    def preprocess_text(self, text):
        """Полный пайплайн предобработки"""
        if not text or not isinstance(text, str):
            return []
            
        # Очистка
        text = self.clean_text(text)
        if not text:
            return []
        
        # Токенизация
        if self.tokenization_method == "whitespace":
            tokens = self.tokenize_whitespace(text)
        elif self.tokenization_method == "regex":
            tokens = self.tokenize_regex(text)
        elif self.tokenization_method == "bpe":
            tokens = self.tokenize_bpe(text)
        else:
            tokens = self.tokenize_whitespace(text)
        
        # Фильтрация по длине
        min_len = PREPROCESSING_CONFIG["min_token_length"]
        max_len = PREPROCESSING_CONFIG["max_token_length"]
        tokens = [t for t in tokens if min_len <= len(t) <= max_len]
        
        # Лемматизация
        if PREPROCESSING_CONFIG["lemmatization"] and tokens:
            tokens = self.lemmatize(tokens)
        
        return tokens

class SimpleBPEProcessor:
    """Простая реализация BPE процессора"""
    def __init__(self):
        self.bpe_codes = {}
    
    def load_model(self, model_path):
        """Загрузка BPE модели из файла"""
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ' ' in line:
                        token, count = line.strip().rsplit(' ', 1)
                        self.bpe_codes[token] = int(count)
        except Exception as e:
            print(f"Error loading BPE model: {e}")
    
    def encode(self, text):
        """Простое кодирование текста с использованием BPE"""
        # Упрощенная реализация - в реальности нужен полный алгоритм BPE
        words = text.split()
        encoded_words = []
        for word in words:
            # Базовая имитация BPE - разбиение на символы с соединением частых пар
            if len(word) > 10 and word in self.bpe_codes:
                encoded_words.append(word)
            else:
                # Просто разбиваем на символы для демонстрации
                encoded_words.extend([c for c in word])
        return encoded_words

# Утилита для обработки множества текстов
def preprocess_texts(texts, tokenization_method="whitespace", bpe_model_path=None):
    """Предобработка списка текстов"""
    preprocessor = TextPreprocessor(tokenization_method=tokenization_method)
    
    if tokenization_method == "bpe" and bpe_model_path:
        preprocessor.load_bpe_model(bpe_model_path)
    
    processed_texts = []
    for text in texts:
        tokens = preprocessor.preprocess_text(text)
        processed_texts.append(tokens)
    
    return processed_texts