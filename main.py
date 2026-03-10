# -*- coding: utf-8 -*-
"""
ФИНАЛЬНЫЙ ПРОЕКТ: ЧАТ-БОТ С ЛОКАЛЬНОЙ LLM И КЛАССИФИКАТОРАМИ
Автор: Арина Коротченко

Основные возможности:
1. Моральный фильтр (классификатор на основе ключевых слов)
2. Классификатор функций (определяет, какую функцию запрашивает пользователь)
3. Языковой классификатор (русский, английский, немецкий, французский)
4. Работа с пользовательскими базами данных (RAG)
5. Извлечение табличных данных
6. Пересказ и генерация тегов
7. Решение задач с пошаговым выводом
"""

import json
import requests
import chromadb
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
from collections import Counter
import pickle

# ============================================================================
# 1. PYDANTIC МОДЕЛИ ДЛЯ СТРУКТУРИРОВАННОГО ВЫВОДА
# ============================================================================

class StructuredSolution(BaseModel):
    """Структурированное решение задачи"""
    problem: str
    steps: List[str] = Field(..., description="Пошаговое решение")
    final_answer: str
    explanation: str

class TableData(BaseModel):
    """Структурированные табличные данные"""
    headers: List[str]
    rows: List[List[str]]
    title: Optional[str] = None

class TextSummary(BaseModel):
    """Результат пересказа текста"""
    summary: str
    tags: List[str] = Field(..., description="Тематические теги")
    key_phrases: List[str] = Field(..., description="Ключевые фразы")

class InstitutionRow(BaseModel):
    """Строка таблицы научных учреждений"""
    name: str
    field_of_science: str
    location: str

# ============================================================================
# 2. КЛАССИФИКАТОРЫ
# ============================================================================

class MoralClassifier:
    """
    Моральный классификатор на основе ключевых слов
    (в реальном проекте можно заменить на ML-модель)
    """
    
    # Запрещенные категории
    FORBIDDEN_CATEGORIES = {
        'weapons': [
            'пистолет', 'пулемет', 'автомат', 'винтовка', 'револьвер', 'огнестрел',
            'оружие', 'взрывчатка', 'динамит', 'тротил', 'бомба', 'граната',
            'мина', 'снаряд', 'патрон', 'гильза', 'взрывать', 'стрелять'
        ],
        'drugs': [
            'наркотик', 'героин', 'кокаин', 'амфетамин', 'марихуана', 'гашиш',
            'лсд', 'экстази', 'метадон', 'опиум', 'морфин', 'инъекция'
        ],
        'violence': [
            'убить', 'убийство', 'насилие', 'пытка', 'истязать', 'мучить',
            'избить', 'зарезать', 'застрелить', 'взорвать', 'терроризм'
        ],
        'illegal': [
            'взлом', 'хакерство', 'кража', 'ограбление', 'мошенничество',
            'подделка', 'контрабанда', 'коррупция', 'взятка', 'рэкет'
        ],
        'dangerous_instructions': [
            'как сделать', 'инструкция по', 'самодельный', 'в домашних условиях',
            'собрать', 'изготовить', 'производство', 'рецепт изготовления'
        ]
    }
    
    # Мультиязычные запрещенные слова
    MULTILINGUAL_FORBIDDEN = {
        'en': [
            'gun', 'rifle', 'pistol', 'weapon', 'explosive', 'bomb', 'drug',
            'cocaine', 'heroin', 'kill', 'murder', 'violence', 'torture',
            'hack', 'steal', 'rob', 'fraud', 'how to make', 'instructions'
        ],
        'de': [
            'pistole', 'gewehr', 'waffe', 'sprengstoff', 'bombe', 'droge',
            'kokain', 'heroin', 'töten', 'mord', 'gewalt', 'folter',
            'hacken', 'stehlen', 'betrug', 'anleitung'
        ],
        'fr': [
            'pistolet', 'fusil', 'arme', 'explosif', 'bombe', 'drogue',
            'cocaïne', 'héroïne', 'tuer', 'meurtre', 'violence', 'torture',
            'pirater', 'voler', 'fraude', 'manuel'
        ]
    }
    
    def __init__(self):
        self.threshold = 0.3  # Порог для классификации
    
    def is_safe(self, text: str, lang: str = 'ru') -> Tuple[bool, str, float]:
        """
        Проверяет безопасность текста
        Возвращает: (is_safe, reason, confidence)
        """
        text_lower = text.lower()
        danger_score = 0
        reason = ""
        
        # Проверка по категориям
        for category, keywords in self.FORBIDDEN_CATEGORIES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    danger_score += 1
                    if not reason:
                        reason = f"Обнаружено запрещенное слово из категории '{category}'"
        
        # Проверка по мультиязычным словам
        if lang in self.MULTILINGUAL_FORBIDDEN:
            for keyword in self.MULTILINGUAL_FORBIDDEN[lang]:
                if keyword in text_lower:
                    danger_score += 1
                    if not reason:
                        reason = f"Обнаружено запрещенное слово на {lang}"
        
        # Проверка опасных комбинаций
        has_dangerous_instruction = any(
            phrase in text_lower for phrase in self.FORBIDDEN_CATEGORIES['dangerous_instructions']
        )
        
        has_dangerous_topic = any(
            keyword in text_lower 
            for category in ['weapons', 'drugs', 'violence'] 
            for keyword in self.FORBIDDEN_CATEGORIES[category]
        )
        
        if has_dangerous_instruction and has_dangerous_topic:
            danger_score += 5
            reason = "Опасная комбинация: инструкция по созданию опасных предметов"
        
        # Нормализация скора
        words_count = len(text_lower.split())
        normalized_score = danger_score / max(words_count, 1)
        
        is_safe = normalized_score < self.threshold
        
        if not reason and not is_safe:
            reason = "Запрос содержит подозрительные паттерны"
        
        return is_safe, reason, normalized_score

class FunctionClassifier:
    """Классификатор функций чат-бота"""
    
    # Ключевые слова для каждой функции
    FUNCTION_KEYWORDS = {
        'rag_search': [
            'найди', 'поиск', 'информация', 'данные', 'сведения',
            'кто', 'что', 'где', 'когда', 'почему', 'как',
            'расскажи', 'объясни', 'что такое'
        ],
        'table_extraction': [
            'таблица', 'таблицу', 'список', 'перечень',
            'систематизируй', 'структурируй', 'организуй в таблицу',
            'выведи в виде таблицы'
        ],
        'summarization': [
            'перескажи', 'кратко', 'резюме', 'аннотация',
            'суть', 'основное', 'сократи', 'сжато'
        ],
        'tag_generation': [
            'теги', 'метки', 'ключевые слова',
            'категории', 'тематика', 'тэги'
        ],
        'problem_solving': [
            'реши', 'задача', 'проблема', 'вопрос',
            'вычисли', 'посчитай', 'найди решение',
            'логическая задача', 'математическая задача'
        ],
        'recommendation': [
            'рекомендация', 'посоветуй', 'что почитать',
            'что посмотреть', 'какой выбрать', 'лучший'
        ],
        'database_management': [
            'база данных', 'создать базу', 'добавить в базу',
            'моя база', 'личная база', 'управление базой'
        ]
    }
    
    def __init__(self):
        self.weights = {
            'rag_search': 1.0,
            'table_extraction': 1.2,  # Более высокий вес для специфичных запросов
            'summarization': 1.1,
            'tag_generation': 1.0,
            'problem_solving': 1.3,
            'recommendation': 1.0,
            'database_management': 1.5  # Самый высокий вес для управления БД
        }
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Классифицирует запрос пользователя
        Возвращает: (function_name, confidence)
        """
        text_lower = text.lower()
        scores = {}
        
        for function, keywords in self.FUNCTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            # Учитываем вес функции
            weighted_score = score * self.weights.get(function, 1.0)
            scores[function] = weighted_score
        
        # Если ни одна функция не набрала очков, используем RAG по умолчанию
        if not any(scores.values()):
            return 'rag_search', 0.1
        
        # Находим функцию с максимальным счетом
        best_function = max(scores, key=scores.get)
        max_score = scores[best_function]
        
        # Нормализуем уверенность
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.1
        
        return best_function, confidence

class LanguageClassifier:
    """Классификатор языка текста"""
    
    # Языковые паттерны
    LANGUAGE_PATTERNS = {
        'ru': {
            'alphabet': r'[а-яё]',
            'common_words': ['и', 'в', 'не', 'на', 'я', 'ты', 'он', 'мы', 'вы', 'они'],
            'unique': ['ё', 'ы', 'э', 'ъ']
        },
        'en': {
            'alphabet': r'[a-z]',
            'common_words': ['the', 'and', 'you', 'that', 'have', 'for', 'with', 'this'],
            'unique': []
        },
        'de': {
            'alphabet': r'[a-zäöüß]',
            'common_words': ['und', 'die', 'der', 'das', 'ist', 'ich', 'du', 'wir'],
            'unique': ['ä', 'ö', 'ü', 'ß']
        },
        'fr': {
            'alphabet': r'[a-zàâçéèêëîïôûùüÿæœ]',
            'common_words': ['et', 'le', 'la', 'les', 'un', 'une', 'je', 'tu', 'il'],
            'unique': ['à', 'â', 'ç', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'û', 'ù', 'ü', 'ÿ', 'æ', 'œ']
        }
    }
    
    def detect(self, text: str) -> str:
        """
        Определяет язык текста
        Возвращает код языка: 'ru', 'en', 'de', 'fr'
        """
        text_lower = text.lower()
        
        # Проверка уникальных символов
        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            for unique_char in patterns['unique']:
                if unique_char in text_lower:
                    return lang_code
        
        # Подсчет вероятностей по алфавиту
        lang_scores = {}
        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            # Проверяем соответствие алфавиту
            non_lang_chars = re.findall(r'[^\s\w]', text_lower)  # Символы не из алфавита
            text_without_special = re.sub(r'[^\s\w]', '', text_lower)
            
            if patterns['alphabet']:
                lang_chars = re.findall(patterns['alphabet'], text_without_special, re.IGNORECASE)
                score = len(lang_chars) / max(len(text_without_special), 1)
                lang_scores[lang_code] = score
        
        # Если есть явный лидер
        if lang_scores:
            best_lang = max(lang_scores, key=lang_scores.get)
            if lang_scores[best_lang] > 0.5:
                return best_lang
        
        # Проверка по общим словам
        word_scores = {}
        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for word in patterns['common_words']:
                if word in text_lower.split():
                    score += 1
            word_scores[lang_code] = score
        
        if any(word_scores.values()):
            return max(word_scores, key=word_scores.get)
        
        # По умолчанию - английский
        return 'en'

# ============================================================================
# 3. RAG СИСТЕМА И УПРАВЛЕНИЕ БАЗАМИ ДАННЫХ
# ============================================================================

class SimpleSearchRAG:
    """Простая RAG система на основе текстового поиска (как в chatbot.py)"""
    
    def __init__(self):
        self.all_documents = []
        self.collection = None
        self.client = chromadb.Client()
    
    def create_collection(self, base_documents, user_documents):
        """Создает коллекцию документов для поиска"""
        self.all_documents = base_documents + user_documents
        if not self.all_documents:
            return None
        
        documents, metadatas, ids = [], [], []
        for i, doc in enumerate(self.all_documents):
            # Формируем текст для индексации
            text = f"Название: {doc.get('title', '')}. Содержание: {doc.get('content', '')}. "
            if 'author' in doc:
                text += f"Автор: {doc.get('author', '')}. "
            if 'category' in doc:
                text += f"Категория: {doc.get('category', '')}. "
            if 'year' in doc:
                text += f"Год: {doc.get('year', '')}. "
            
            documents.append(text)
            metadatas.append(doc)
            ids.append(str(i))
        
        # Создаем или получаем коллекцию
        try:
            self.collection = self.client.get_or_create_collection("documents")
            # Очищаем коллекцию перед добавлением новых документов
            self.collection.delete(ids=self.collection.get()['ids'])
        except:
            self.collection = self.client.create_collection("documents")
        
        # Добавляем документы
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return self.collection
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Поиск документов по запросу"""
        if not self.collection or not self.all_documents:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            return formatted_results
        except Exception as e:
            print(f"[ERROR] Ошибка поиска: {e}")
            return []
    
    def extract_answer_from_context(self, context: str, query: str, lang: str = 'ru') -> str:
        """Извлекает ответ из контекста (упрощенная версия из chatbot.py)"""
        try:
            # Разбиваем контекст на предложения
            lines = context.split(". ")
            metadata = {}
            
            # Извлекаем информацию из структурированного текста
            for line in lines:
                if line.startswith("Название:"):
                    metadata["title"] = line.replace("Название:", "").strip()
                elif line.startswith("Содержание:"):
                    metadata["content"] = line.replace("Содержание:", "").strip()
                elif line.startswith("Автор:"):
                    metadata["author"] = line.replace("Автор:", "").strip()
                elif line.startswith("Категория:"):
                    metadata["category"] = line.replace("Категория:", "").strip()
            
            # Формируем ответ
            if "content" in metadata:
                # Берем первые 2 предложения из содержания
                sentences = re.split(r'[.!?]+', metadata["content"])
                sentences = [s.strip() for s in sentences if s.strip()]
                summary = ". ".join(sentences[:2]) + "." if sentences else metadata["content"][:200] + "..."
                
                response = f"{metadata.get('title', 'Информация')}: {summary}"
                
                # Добавляем дополнительные детали
                if "author" in metadata:
                    response += f" Автор: {metadata['author']}."
                if "category" in metadata:
                    response += f" Категория: {metadata['category']}."
                
                # Проверяем релевантность
                query_words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 3]
                context_lower = context.lower()
                
                if any(word in context_lower for word in query_words):
                    return response
                else:
                    return "Информация по вашему запросу не найдена."
            else:
                return "Не удалось извлечь информацию."
                
        except Exception as e:
            print(f"[ERROR] Ошибка при извлечении ответа: {e}")
            return "Ошибка при обработке контекста."

class DatabaseManager:
    """Менеджер пользовательских баз данных"""
    
    def __init__(self, storage_path: str = "./user_databases"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Инициализируем простую RAG систему
        self.rag_system = SimpleSearchRAG()
        
        # Загружаем метаданные баз данных
        self.metadata_file = self.storage_path / "databases_meta.json"
        self.databases_meta = self._load_metadata()
        self.current_documents = []
    
    def _load_metadata(self) -> Dict:
        """Загружает метаданные баз данных"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Сохраняет метаданные баз данных"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.databases_meta, f, ensure_ascii=False, indent=2)
    
    def create_database(self, name: str, description: str = "") -> bool:
        """Создает новую базу данных"""
        try:
            # Сохраняем метаданные
            self.databases_meta[name] = {
                'description': description,
                'created_at': datetime.now().isoformat(),
                'document_count': 0,
                'last_updated': datetime.now().isoformat()
            }
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Ошибка при создании БД: {e}")
            return False
    
    def add_document(self, db_name: str, document_id: str, content: str, metadata: Dict = None) -> bool:
        """Добавляет документ в базу данных"""
        try:
            # Создаем структурированный документ
            doc = {
                'id': document_id,
                'content': content,
                'title': metadata.get('title', document_id) if metadata else document_id
            }
            
            # Добавляем дополнительные поля из metadata
            if metadata:
                for key, value in metadata.items():
                    if key != 'title':  # title уже добавлен
                        doc[key] = value
            
            self.current_documents.append(doc)
            
            # Обновляем метаданные
            if db_name in self.databases_meta:
                self.databases_meta[db_name]['document_count'] += 1
                self.databases_meta[db_name]['last_updated'] = datetime.now().isoformat()
                self._save_metadata()
            
            # Перестраиваем RAG индекс
            self.rag_system.create_collection(self.current_documents, [])
            
            return True
        except Exception as e:
            print(f"Ошибка при добавлении документа: {e}")
            return False
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Поиск в базе данных"""
        return self.rag_system.search(query, n_results)
    
    def extract_answer(self, query: str, lang: str = 'ru') -> str:
        """Извлекает ответ на основе поиска"""
        results = self.search(query, n_results=3)
        
        if not results:
            return "Информация не найдена."
        
        # Используем первый результат
        context = results[0]['content']
        return self.rag_system.extract_answer_from_context(context, query, lang)
    
    def list_databases(self) -> List[Dict]:
        """Список всех баз данных"""
        databases = []
        
        for name, meta in self.databases_meta.items():
            databases.append({
                'name': name,
                'description': meta.get('description', ''),
                'document_count': meta.get('document_count', 0),
                'created_at': meta.get('created_at', ''),
                'last_updated': meta.get('last_updated', '')
            })
        
        return databases

# ============================================================================
# 4. ТЕКСТОВАЯ ОБРАБОТКА И АНАЛИЗ
# ============================================================================

class TextProcessor:
    """Обработчик текста для пересказа и генерации тегов"""
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10, lang: str = 'ru') -> List[str]:
        """Извлекает ключевые слова из текста"""
        # Удаляем пунктуацию и приводим к нижнему регистру
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        # Стоп-слова для разных языков
        stopwords = {
            'ru': {'и', 'в', 'не', 'на', 'я', 'ты', 'он', 'мы', 'вы', 'они', 'что', 'это', 'как', 'для'},
            'en': {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'de': {'und', 'die', 'der', 'das', 'in', 'von', 'zu', 'den', 'mit', 'sich', 'des'},
            'fr': {'et', 'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'pour'}
        }
        
        # Фильтруем стоп-слова и короткие слова
        lang_stopwords = stopwords.get(lang, stopwords['en'])
        filtered_words = [w for w in words if w not in lang_stopwords and len(w) > 3]
        
        # Подсчитываем частоту
        word_freq = Counter(filtered_words)
        
        # Возвращаем N самых частых слов
        return [word for word, _ in word_freq.most_common(top_n)]
    
    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        """Создает краткий пересказ текста"""
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Берем первые N предложений
        n_sentences = min(max_sentences, len(sentences))
        return ' '.join(sentences[:n_sentences])
    
    @staticmethod
    def extract_phrases(text: str, min_words: int = 2, max_words: int = 4) -> List[str]:
        """Извлекает значимые словосочетания"""
        words = text.lower().split()
        phrases = []
        
        for i in range(len(words) - min_words + 1):
            for j in range(min_words, min(max_words + 1, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+j])
                # Проверяем, что фраза не состоит только из стоп-слов
                if len(phrase) > 5:  # Минимальная длина
                    phrases.append(phrase)
        
        # Подсчитываем частоту
        phrase_freq = Counter(phrases)
        
        # Возвращаем самые частые фразы
        return [phrase for phrase, _ in phrase_freq.most_common(10)]

# ============================================================================
# 5. ОСНОВНОЙ КЛАСС ЧАТ-БОТА
# ============================================================================

class ChatBot:
    """Главный класс чат-бота"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "mistral"):
        self.ollama_host = ollama_host
        self.model = model
        
        # Инициализируем классификаторы
        self.moral_classifier = MoralClassifier()
        self.function_classifier = FunctionClassifier()
        self.language_classifier = LanguageClassifier()
        
        # Инициализируем менеджер БД
        self.db_manager = DatabaseManager()
        
        # Инициализируем процессор текста
        self.text_processor = TextProcessor()
        
        # Текущий язык и активная БД
        self.current_language = 'ru'
        self.current_database = 'science'
        
        # Создаем демонстрационные базы данных при первом запуске
        self._init_demo_databases()
    
    def _init_demo_databases(self):
        """Создает демонстрационные базы данных"""
        demo_documents = [
            {
                'id': 'science_1',
                'title': 'Черные дыры',
                'content': 'Черная дыра — область пространства-времени, гравитационное притяжение которой настолько велико, что покинуть её не могут даже объекты, движущиеся со скоростью света, в том числе кванты самого света. Черные дыры возникают в результате гравитационного коллапса массивных звезд.',
                'category': 'астрономия',
                'author': 'Научный журнал'
            },
            {
                'id': 'science_2',
                'title': 'Квантовая механика',
                'content': 'Квантовая механика — фундаментальная теория в физике, описывающая поведение материи и энергии на уровне атомов и субатомных частиц. Она вводит понятие квантовых состояний, которые описываются волновой функцией.',
                'category': 'физика',
                'author': 'Научный журнал'
            },
            {
                'id': 'science_3',
                'title': 'ДНК и генетика',
                'content': 'ДНК (дезоксирибонуклеиновая кислота) — молекула, хранящая генетическую информацию у всех живых организмов. Структура ДНК была открыта Джеймсом Уотсоном и Фрэнсисом Криком в 1953 году.',
                'category': 'биология',
                'author': 'Научный журнал'
            },
            {
                'id': 'literature_1',
                'title': 'Преступление и наказание',
                'content': 'Роман "Преступление и наказание" Ф.М. Достоевского был написан в 1866 году. Главный герой — Родион Раскольников, бывший студент, который совершает убийство старухи-процентщицы, чтобы проверить свою теорию о "тварях дрожащих" и "право имеющих".',
                'category': 'литература',
                'author': 'Ф.М. Достоевский',
                'year': 1866
            },
            {
                'id': 'history_1',
                'title': 'Великая Отечественная война',
                'content': 'Великая Отечественная война началась 22 июня 1941 года и продолжалась до 9 мая 1945 года. Это была крупнейшая война в истории человечества, в которой участвовали десятки стран.',
                'category': 'история',
                'author': 'Исторический журнал'
            }
        ]
        
        # Создаем базу данных
        self.db_manager.create_database('science', 'Научные статьи и открытия')
        self.db_manager.create_database('literature', 'Литературные произведения')
        self.db_manager.create_database('history', 'Исторические события')
        
        # Добавляем все документы
        for doc in demo_documents:
            metadata = {k: v for k, v in doc.items() if k not in ['id', 'content']}
            self.db_manager.add_document('science', doc['id'], doc['content'], metadata)
        
        print(f"[INFO] Создано демонстрационных документов: {len(demo_documents)}")
    
    def _get_ollama_response(self, prompt: str, system_prompt: str = None, json_format: bool = False) -> str:
        """Получает ответ от Ollama"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3}
            }
            
            if json_format:
                payload["format"] = "json"
            
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=2000
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"[ERROR] HTTP {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return "[ERROR] Ollama недоступна. Запустите: ollama serve"
        except Exception as e:
            return f"[ERROR] {str(e)}"
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Основной метод обработки запроса пользователя
        Возвращает словарь с результатом
        """
        result = {
            'success': False,
            'response': '',
            'function_used': '',
            'language': 'ru',
            'is_safe': True,
            'confidence': 0.0
        }
        
        # 1. Определяем язык
        detected_lang = self.language_classifier.detect(user_query)
        self.current_language = detected_lang
        result['language'] = detected_lang
        
        # 2. Проверяем моральную безопасность
        is_safe, reason, confidence = self.moral_classifier.is_safe(user_query, detected_lang)
        result['is_safe'] = is_safe
        result['confidence'] = confidence
        
        if not is_safe:
            rejection_messages = {
                'ru': f"К сожалению, ваш запрос нарушает мои моральные принципы и не будет обработан. ({reason})",
                'en': f"Unfortunately, your request violates my ethical principles and will not be processed. ({reason})",
                'de': f"Leider verstößt Ihre Anfrage gegen meine ethischen Grundsätze und wird nicht verarbeitet. ({reason})",
                'fr': f"Malheureusement, votre demande enfreint mes principes éthiques et ne sera pas traitée. ({reason})"
            }
            result['response'] = rejection_messages.get(detected_lang, rejection_messages['ru'])
            return result
        
        # 3. Определяем функцию
        function, func_confidence = self.function_classifier.classify(user_query)
        result['function_used'] = function
        
        # 4. Обрабатываем запрос в зависимости от функции
        if function == 'rag_search':
            response = self._handle_rag_search(user_query)
        elif function == 'table_extraction':
            response = self._handle_table_extraction(user_query)
        elif function == 'summarization':
            response = self._handle_summarization(user_query)
        elif function == 'tag_generation':
            response = self._handle_tag_generation(user_query)
        elif function == 'problem_solving':
            response = self._handle_problem_solving(user_query)
        elif function == 'recommendation':
            response = self._handle_recommendation(user_query)
        elif function == 'database_management':
            response = self._handle_database_management(user_query)
        else:
            response = self._handle_rag_search(user_query)
        
        result['response'] = response
        result['success'] = True
        
        return result
    
    def _handle_rag_search(self, query: str) -> str:
        """Обработка RAG-запросов - УПРОЩЕННАЯ ВЕРСИЯ КАК В CHATBOT.PY"""
        print(f"[DEBUG] Поиск по запросу: {query}")
        
        # Используем простой поиск из DatabaseManager
        answer = self.db_manager.extract_answer(query, self.current_language)
        
        if answer == "Информация не найдена.":
            # Попробуем найти вручную по ключевым словам
            query_lower = query.lower()
            
            # Проверяем наличие ключевых слов в документах
            for doc in self.db_manager.current_documents:
                content_lower = doc.get('content', '').lower()
                title_lower = doc.get('title', '').lower()
                
                # Ищем совпадения ключевых слов
                query_words = [w for w in re.findall(r'\w+', query_lower) if len(w) > 3]
                found = False
                
                for word in query_words:
                    if word in content_lower or word in title_lower:
                        found = True
                        break
                
                if found:
                    # Возвращаем найденный документ
                    title = doc.get('title', 'Без названия')
                    content = doc.get('content', '')
                    
                    # Формируем краткий ответ
                    sentences = re.split(r'[.!?]+', content)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    summary = ". ".join(sentences[:2]) + "." if sentences else content[:200] + "..."
                    
                    return f"{title}: {summary}"
            
            return self._get_response_in_language("Информация не найдена.")
        
        return answer
    
    def _handle_table_extraction(self, query: str) -> str:
        """Извлечение табличных данных"""
        # Пытаемся понять, извлекать ли таблицу из БД или из текста
        if "баз" in query.lower() or "баз" in query:
            # Извлекаем из БД
            results = self.db_manager.search(query, n_results=10)
            
            if not results:
                return self._get_response_in_language("В базе данных не найдено информации для создания таблицы.")
            
            # Создаем таблицу из результатов
            table_content = "| Заголовок | Содержание |\n|---|---|\n"
            for i, result in enumerate(results[:5], 1):
                metadata = result.get('metadata', {})
                title = metadata.get('title', f'Документ {i}')
                content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                table_content += f"| {title} | {content} |\n"
            
            return table_content
        
        else:
            # Извлекаем таблицу из запроса с помощью LLM
            system_prompt = """Ты - эксперт по структурированию данных. 
            Создай таблицу на основе предоставленного текста.
            Определи наиболее подходящие колонки.
            Верни результат в формате Markdown таблицы."""
            
            prompt = f"""Создай структурированную таблицу на основе следующего текста:

{query}

Верни ТОЛЬКО таблицу в формате Markdown."""
            
            return self._get_ollama_response(prompt, system_prompt)
    
    def _handle_summarization(self, query: str) -> str:
        """Создание пересказа"""
        # Если запрос короткий, считаем его текстом для пересказа
        if len(query.split()) > 5:
            text_to_summarize = query
        else:
            # Ищем в БД
            results = self.db_manager.search(query, n_results=1)
            if results:
                text_to_summarize = results[0]['content']
            else:
                return self._get_response_in_language("Не найдено текста для пересказа.")
        
        # Создаем пересказ
        summary = self.text_processor.summarize_text(text_to_summarize)
        
        # Улучшаем через LLM
        system_prompt = self._get_system_prompt("summarizer")
        prompt = f"""Создай краткий пересказ следующего текста:

{text_to_summarize}

Пересказ должен быть в 2-3 предложениях."""
        
        llm_summary = self._get_ollama_response(prompt, system_prompt)
        
        return f"**Краткий пересказ:**\n{llm_summary}"
    
    def _handle_tag_generation(self, query: str) -> str:
        """Генерация тегов"""
        # Анализируем текст
        keywords = self.text_processor.extract_keywords(query, top_n=10, lang=self.current_language)
        phrases = self.text_processor.extract_phrases(query)
        
        # Используем LLM для улучшения тегов
        system_prompt = self._get_system_prompt("tag_generator")
        prompt = f"""Проанализируй текст и предложи 5-7 наиболее релевантных тегов:

{query}

Теги должны быть краткими и информативными. Верни список тегов через запятую."""
        
        llm_tags = self._get_ollama_response(prompt, system_prompt)
        
        response = f"**Ключевые слова:** {', '.join(keywords[:5])}\n\n"
        response += f"**Значимые фразы:** {', '.join(phrases[:3])}\n\n"
        response += f"**Тематические теги (LLM):** {llm_tags}"
        
        return response
    
    def _handle_problem_solving(self, query: str) -> str:
        """Решение задач с пошаговым выводом"""
        system_prompt = """Ты - эксперт по решению задач. 
        Решай задачи пошагово, показывая все этапы рассуждения.
        Формат ответа:
        Шаг 1: [Описание первого шага]
        Шаг 2: [Описание второго шага]
        ...
        Ответ: [Финальный ответ]"""
        
        prompt = f"""Реши следующую задачу пошагово:

{query}

Покажи все промежуточные вычисления и рассуждения."""
        
        return self._get_ollama_response(prompt, system_prompt)
    
    def _handle_recommendation(self, query: str) -> str:
        """Генерация рекомендаций"""
        # Ищем похожие документы
        results = self.db_manager.search(query, n_results=3)
        
        if not results:
            return self._get_response_in_language(f"В базе данных не найдено материалов для рекомендаций.")
        
        # Формируем контекст для рекомендации
        recommendations = []
        for i, result in enumerate(results[:3]):
            metadata = result.get('metadata', {})
            title = metadata.get('title', f'Материал {i+1}')
            content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
            recommendations.append(f"{i+1}. {title}: {content}")
        
        return f"**Рекомендации:**\n" + "\n".join(recommendations)
    
    def _handle_database_management(self, query: str) -> str:
        """Управление базами данных"""
        # Простые команды для управления БД
        query_lower = query.lower()
        
        # Список БД
        if "список" in query_lower or "list" in query_lower:
            databases = self.db_manager.list_databases()
            if not databases:
                return self._get_response_in_language("Нет созданных баз данных.")
            
            response = "**Доступные базы данных:**\n"
            for db in databases:
                response += f"- {db['name']}: {db['description']} ({db['document_count']} документов)\n"
            return response
        
        # Создание БД
        elif "создай" in query_lower or "create" in query_lower:
            # Извлекаем название БД из запроса
            match = re.search(r'(?:создай|create)[\s]+(?:базу|database)[\s]+(\w+)', query_lower)
            if match:
                db_name = match.group(1)
                if self.db_manager.create_database(db_name, f"База данных '{db_name}'"):
                    self.current_database = db_name
                    return self._get_response_in_language(f"База данных '{db_name}' успешно создана и выбрана!")
                else:
                    return self._get_response_in_language(f"Ошибка при создании базы данных '{db_name}'.")
            else:
                return self._get_response_in_language("Укажите название базы данных. Формат: 'создай базу название_базы'")
        
        # Выбор БД
        elif "выбери" in query_lower or "select" in query_lower:
            match = re.search(r'(?:выбери|select)[\s]+(?:базу|database)[\s]+(\w+)', query_lower)
            if match:
                db_name = match.group(1)
                databases = [db['name'] for db in self.db_manager.list_databases()]
                if db_name in databases:
                    self.current_database = db_name
                    return self._get_response_in_language(f"Выбрана база данных '{db_name}'.")
                else:
                    return self._get_response_in_language(f"База данных '{db_name}' не найдена.")
        
        return self._get_response_in_language("""
Доступные команды управления БД:
- "список баз" - показать все базы данных
- "создай базу название" - создать новую БД
- "выбери базу название" - выбрать активную БД
- "добавь в базу текст" - добавить документ (в разработке)
""")
    
    def _get_system_prompt(self, role: str) -> str:
        """Возвращает системный промт для роли на текущем языке"""
        prompts = {
            'rag_assistant': {
                'ru': 'Ты - помощник, отвечающий на вопросы на основе предоставленного контекста.',
                'en': 'You are an assistant answering questions based on the provided context.',
                'de': 'Sie sind ein Assistent, der Fragen auf der Grundlage des bereitgestellten Kontexts beantwortet.',
                'fr': 'Vous êtes un assistant qui répond aux questions sur la base du contexte fourni.'
            },
            'summarizer': {
                'ru': 'Ты - эксперт по созданию кратких пересказов текстов.',
                'en': 'You are an expert at creating brief summaries of texts.',
                'de': 'Sie sind Experte für die Erstellung kurzer Textzusammenfassungen.',
                'fr': 'Vous êtes un expert dans la création de résumés brefs de textes.'
            },
            'tag_generator': {
                'ru': 'Ты - эксперт по анализу текстов и выделению ключевых тегов.',
                'en': 'You are an expert at analyzing texts and extracting key tags.',
                'de': 'Sie sind Experte für die Textanalyse und die Extraktion von Schlüssel-Tags.',
                'fr': 'Vous êtes un expert dans l\'analyse de textes et l\'extraction de mots-clés.'
            },
            'recommender': {
                'ru': 'Ты - эксперт по рекомендациям, помогающий выбрать наиболее подходящие материалы.',
                'en': 'You are a recommendation expert helping to choose the most suitable materials.',
                'de': 'Sie sind ein Empfehlungsexperte, der bei der Auswahl der am besten geeigneten Materialien hilft.',
                'fr': 'Vous êtes un expert en recommandation qui aide à choisir les matériaux les plus appropriés.'
            }
        }
        
        role_prompts = prompts.get(role, prompts['rag_assistant'])
        return role_prompts.get(self.current_language, role_prompts['ru'])
    
    def _get_response_in_language(self, message: str) -> str:
        """Возвращает стандартные сообщения на текущем языке"""
        responses = {
            'ru': message,
            'en': {
                'База данных не выбрана. Пожалуйста, создайте или выберите БД.': 'Database not selected. Please create or select a database.',
                'В базе данных не найдено информации по вашему запросу.': 'No information found in the database for your query.',
                'Сначала выберите базу данных.': 'First select a database.',
                'В базе данных не найдено информации для создания таблицы.': 'No information found in the database to create a table.',
                'Не найдено текста для пересказа.': 'No text found for summarization.',
                'Сначала выберите базу данных для рекомендаций.': 'First select a database for recommendations.',
                'В базе данных не найдено материалов для рекомендаций.': 'No materials found in the database for recommendations.',
                'Нет созданных баз данных.': 'No databases created.',
                'База данных успешно создана и выбрана!': 'Database successfully created and selected!',
                'Ошибка при создании базы данных.': 'Error creating database.',
                'Укажите название базы данных. Формат: \'создай базу название_базы\'': 'Specify the database name. Format: \'create database name\'',
                'Выбрана база данных.': 'Database selected.',
                'База данных не найдена.': 'Database not found.',
                'Информация не найдена.': 'Information not found.'
            }.get(message, message),
            'de': {
                'База данных не выбрана. Пожалуйста, создайте или выберите БД.': 'Datenbank nicht ausgewählt. Bitte erstellen oder wählen Sie eine Datenbank.',
                'В базе данных не найдено информации по вашему запросу.': 'Keine Informationen zu Ihrer Anfrage in der Datenbank gefunden.',
                'Сначала выберите базу данных.': 'Wählen Sie zuerst eine Datenbank.',
                'В базе данных не найдено информации для создания таблицы.': 'Keine Informationen in der Datenbank zum Erstellen einer Tabelle gefunden.',
                'Не найдено текста для пересказа.': 'Kein Text zur Zusammenfassung gefunden.',
                'Сначала выберите базу данных для рекомендаций.': 'Wählen Sie zuerst eine Datenbank für Empfehlungen.',
                'В базе данных не найдено материалов для рекомендаций.': 'Keine Materialien für Empfehlungen in der Datenbank gefunden.',
                'Нет созданных баз данных.': 'Keine Datenbanken erstellt.',
                'База данных успешно создана и выбрана!': 'Datenbank erfolgreich erstellt und ausgewählt!',
                'Ошибка при создании базы данных.': 'Fehler beim Erstellen der Datenbank.',
                'Укажите название базы данных. Формат: \'создай базу название_базы\'': 'Geben Sie den Datenbanknamen an. Format: \'erstelle Datenbank name\'',
                'Выбрана база данных.': 'Datenbank ausgewählt.',
                'База данных не найдена.': 'Datenbank nicht gefunden.',
                'Информация не найдена.': 'Informationen nicht gefunden.'
            }.get(message, message),
            'fr': {
                'База данных не выбрана. Пожалуйста, создайте или выберите БД.': 'Base de données non sélectionnée. Veuillez créer ou sélectionner une base de données.',
                'В базе данных не найдено информации по вашему запросу.': 'Aucune information trouvée dans la base de données pour votre requête.',
                'Сначала выберите базу данных.': 'Sélectionnez d\'abord une base de данных.',
                'В базе данных не найдено информации для создания таблицы.': 'Aucune information trouvée dans la base de données pour créer un tableau.',
                'Не найдено текста для пересказа.': 'Aucun texte trouvé pour le résumé.',
                'Сначала выберите базу данных для рекомендаций.': 'Sélectionnez d\'abord une base de données pour les recommandations.',
                'В базе данных не найдено материалов для рекомендаций.': 'Aucun matériel trouvé dans la base de données pour les recommandations.',
                'Нет созданных баз данных.': 'Aucune base de données créée.',
                'База данных успешно создана и выбрана!': 'Base de données créée et sélectionnée avec succès!',
                'Ошибка при создании базы данных.': 'Erreur lors de la création de la base de données.',
                'Укажите название базы данных. Формат: \'создай базу название_базы\'': 'Spécifiez le nom de la base de données. Format: \'créez base de données nom\'',
                'Выбрана база данных.': 'Base de données sélectionnée.',
                'База данных не найдена.': 'Base de données non trouvée.',
                'Информация не найдена.': 'Information non trouvée.'
            }.get(message, message)
        }
        
        if self.current_language in responses and isinstance(responses[self.current_language], dict):
            return responses[self.current_language].get(message, message)
        
        return responses.get(self.current_language, message)

# ============================================================================
# 6. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================================================

def print_welcome():
    """Выводит приветственное сообщение"""
    print("=" * 60)
    print("ЧАТ-БОТ С ЛОКАЛЬНОЙ LLM И КЛАССИФИКАТОРАМИ")
    print("=" * 60)
    print()
    print("Основные возможности:")
    print("1. Моральный фильтр (блокирует опасные запросы)")
    print("2. Автоматическое определение функции")
    print("3. Поддержка 4 языков: русский, английский, немецкий, французский")
    print("4. Работа с пользовательскими базами данных (RAG)")
    print("5. Извлечение табличных данных")
    print("6. Пересказ текстов и генерация тегов")
    print("7. Решение задач с пошаговым выводом")
    print()
    print("Команды:")
    print("  /help             - Справка")
    print("  /lang             - Выбрать язык")
    print("  /list_db          - Список баз данных")
    print("  /create_db <имя>  - Создать БД")
    print("  /select_db <имя>  - Выбрать БД")
    print("  /add_doc          - Добавить документ в БД")
    print("  /quit             - Выход")
    print()
    print("Просто введите ваш запрос, и бот автоматически определит,")
    print("какую функцию использовать!")
    print("=" * 60)
    print()

def print_help():
    """Выводит справку"""
    print("\n" + "=" * 60)
    print("СПРАВКА ПО ИСПОЛЬЗОВАНИЮ")
    print("=" * 60)
    print()
    print("1. МОРАЛЬНЫЙ ФИЛЬТР:")
    print("   Бот автоматически проверяет запросы на наличие:")
    print("   - Инструкций по созданию оружия")
    print("   - Рецептов наркотиков")
    print("   - Призывов к насилию")
    print("   - Другого опасного содержимого")
    print()
    print("2. АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ФУНКЦИИ:")
    print("   Просто введите запрос - бот сам поймет, что нужно:")
    print("   - 'Найди информацию о квантовой физике' → RAG-поиск")
    print("   - 'Создай таблицу из этого текста' → Извлечение таблицы")
    print("   - 'Перескажи текст про космос' → Пересказ")
    print("   - 'Какие теги у этого текста?' → Генерация тегов")
    print("   - 'Реши задачу: 2x + 5 = 15' → Решение задачи")
    print()
    print("3. РАБОТА С БАЗАМИ ДАННЫХ:")
    print("   /create_db my_books - Создать БД 'my_books'")
    print("   /select_db science  - Выбрать БД 'science'")
    print("   /add_doc           - Добавить документ в БД")
    print()
    print("4. ПРИМЕРЫ ЗАПРОСОВ:")
    print("   - 'Что такое черные дыры?'")
    print("   - 'Расскажи о квантовой механике'")
    print("   - 'Кто такой Достоевский?'")
    print("   - 'Создай таблицу научных открытий'")
    print("   - 'Перескажи текст про искусственный интеллект'")
    print("   - 'Какие ключевые слова в этом тексте?'")
    print("   - 'Реши уравнение: x^2 - 5x + 6 = 0'")
    print("=" * 60 + "\n")

def main():
    """Главная функция"""
    
    print_welcome()
    
    # Инициализация бота
    print("[INFO] Инициализация чат-бота...")
    try:
        bot = ChatBot()
        print("[INFO]  Бот успешно инициализирован!")
        print(f"[INFO]  Модель: {bot.model}")
        databases = bot.db_manager.list_databases()
        print(f"[INFO]  Доступные БД: {len(databases)}")
        for db in databases:
            print(f"        - {db['name']}: {db['description']} ({db['document_count']} документов)")
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        print("Убедитесь, что Ollama запущена: ollama serve")
        return
    
    print()
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            # Обработка команд
            if user_input.lower() == "/quit":
                print("\nДо свидания!")
                break
            
            elif user_input.lower() == "/help":
                print_help()
                continue
            
            elif user_input.lower() == "/lang":
                print("Выберите язык:")
                print("1. Русский")
                print("2. English")
                print("3. Deutsch")
                print("4. Français")
                choice = input("Ваш выбор (1-4): ").strip()
                lang_map = {"1": "ru", "2": "en", "3": "de", "4": "fr"}
                if choice in lang_map:
                    bot.current_language = lang_map[choice]
                    print(f"Язык установлен: {lang_map[choice]}")
                else:
                    print("Неверный выбор")
                continue
            
            elif user_input.lower() == "/list_db":
                databases = bot.db_manager.list_databases()
                if databases:
                    print("\nДоступные базы данных:")
                    for db in databases:
                        active = " (активна)" if db['name'] == bot.current_database else ""
                        print(f"  - {db['name']}{active}: {db['description']} ({db['document_count']} документов)")
                else:
                    print("Нет доступных баз данных")
                print()
                continue
            
            elif user_input.lower().startswith("/create_db "):
                db_name = user_input[10:].strip()
                if db_name:
                    success = bot.db_manager.create_database(db_name, f"База данных '{db_name}'")
                    if success:
                        bot.current_database = db_name
                        print(f"БД '{db_name}' создана и выбрана!")
                    else:
                        print(f"Ошибка при создании БД '{db_name}'")
                continue
            
            elif user_input.lower().startswith("/select_db "):
                db_name = user_input[10:].strip()
                databases = [db['name'] for db in bot.db_manager.list_databases()]
                if db_name in databases:
                    bot.current_database = db_name
                    print(f"Выбрана БД: {db_name}")
                else:
                    print(f"БД '{db_name}' не найдена")
                continue
            
            elif user_input.lower() == "/add_doc":
                if not bot.current_database:
                    print("Сначала выберите базу данных")
                    continue
                
                print("Добавление документа в БД:", bot.current_database)
                doc_id = input("ID документа: ").strip()
                if not doc_id:
                    doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                print("Введите текст документа (закончите пустой строкой):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                
                content = "\n".join(lines)
                if content:
                    metadata = {
                        'title': doc_id,
                        'added_at': datetime.now().isoformat(),
                        'language': bot.current_language
                    }
                    
                    success = bot.db_manager.add_document(bot.current_database, doc_id, content, metadata)
                    if success:
                        print(f"Документ '{doc_id}' добавлен в БД '{bot.current_database}'")
                    else:
                        print("Ошибка при добавлении документа")
                continue
            
            # Обработка обычного запроса
            print("\n[Обработка...]")
            result = bot.process_query(user_input)
            
            if result['success']:
                print(f"\nБот ({result['function_used']}, {result['language']}):")
                print(result['response'])
            else:
                print("\nБот:", result['response'])
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        
        except Exception as e:
            print(f"\n[ERROR] Ошибка: {e}")
            print("Попробуйте еще раз или введите /quit для выхода\n")

if __name__ == "__main__":
    main()