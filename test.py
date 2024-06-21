from transformers import BertTokenizer, BertModel, MarianMTModel, MarianTokenizer
import torch
import sqlite3
import numpy as np
import logging
from scipy.spatial.distance import cosine, euclidean, cityblock
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
import requests
from lxml import html
import datetime
import asyncio
import urllib.parse
from bs4 import BeautifulSoup
from langdetect import detect
import signal
import os
import sys
import langid
from langdetect import detect, DetectorFactory


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
API_TOKEN = ''
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

ADMIN_USERS = ['', , '', '']
ADMIN_ID = 
DetectorFactory.seed = 0
# Определение состояний для FSM
class Form(StatesGroup):
    group_id = State()
    phrase_id = State()
    phrase = State()
    modality = State()
    intensity = State()
    meaning = State()
    context = State()
    context_code = State()
    approved = State()
    language = State()
    waiting_for_context_choice = State()
    translation_phrase = State()
    translation_language = State()
    user_defined_modality = State()
    user_defined_intensity = State()
    user_defined_meaning = State()
    user_defined_context = State()
    user_defined_context_code = State()
    feedback = State()
    translation_process_choice = State()
    manual_modality = State()
    manual_intensity = State()
    manual_meaning = State()
    manual_context = State()
    manual_context_code = State()
    waiting_for_language_confirmation = State()


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

translation_models = {
    ('ru', 'en'): MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ru-en'),
    ('en', 'ru'): MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ru'),
    ('de', 'en'): MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en'),
    ('en', 'de'): MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
}

translation_tokenizers = {
    ('ru', 'en'): MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en'),
    ('en', 'ru'): MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru'),
    ('de', 'en'): MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en'),
    ('en', 'de'): MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
}

def add_phrase(language, group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved=0):
    phrase = phrase.lower()
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT INTO {table_name} (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id))
    phrase_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Added {language} phrase: {phrase} with group_id: {group_id}, approved: {approved}, id: {phrase_id}, user_id: {user_id}")
    return phrase_id

def translate_to_interlingua(text, src_lang, target_lang='en'):
    try:
        if src_lang != 'en':
            translated_text = translate_text(text, src_lang, 'en')
        else:
            translated_text = text
        return translated_text
    except Exception as e:
        logger.error(f"Translation model error: {e}")
        raise ValueError(f"Translation model error: {e}")

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        return "unknown"

def detect_language_with_langid(text):
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception as e:
        logger.error(f"Error in language detection with langid: {e}")
        return "unknown"

def combined_language_detection(text):
    lang_detect = detect_language(text)
    lang_langid = detect_language_with_langid(text)
    if lang_detect != lang_langid:
        logger.warning(f"Different language detection results: langdetect={lang_detect}, langid={lang_langid}")
        return "conflict"
    return lang_detect

def get_embedding(text):
    text = text.lower()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


def calculate_similarity(vec1, vec2):
    cosine_sim = 1 - cosine(vec1, vec2)
    euclidean_sim = 1 / (1 + euclidean(vec1, vec2))
    manhattan_sim = 1 / (1 + cityblock(vec1, vec2))
    return (cosine_sim + euclidean_sim + manhattan_sim) / 3

# Database functions
def create_tables():
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS russian_phrases (
        id INTEGER PRIMARY KEY,
        group_id INTEGER,
        phrase TEXT NOT NULL,
        modality TEXT,
        intensity REAL,
        meaning TEXT,
        context TEXT,
        context_code INTEGER,
        approved BOOLEAN NOT NULL CHECK (approved IN (0, 1)) DEFAULT 0,
        user_id INTEGER
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS english_phrases (
        id INTEGER PRIMARY KEY,
        group_id INTEGER,
        phrase TEXT NOT NULL,
        modality TEXT,
        intensity REAL,
        meaning TEXT,
        context TEXT,
        context_code INTEGER,
        approved BOOLEAN NOT NULL CHECK (approved IN (0, 1)) DEFAULT 0,
        user_id INTEGER
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS german_phrases (
        id INTEGER PRIMARY KEY,
        group_id INTEGER,
        phrase TEXT NOT NULL,
        modality TEXT,
        intensity REAL,
        meaning TEXT,
        context TEXT,
        context_code INTEGER,
        approved BOOLEAN NOT NULL CHECK (approved IN (0, 1)) DEFAULT 0,
        user_id INTEGER
    )
    ''')
    conn.commit()
    conn.close()





def create_feedback_table():
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT,
        feedback TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()


def get_context_examples(language='en'):
    if language == 'ru':
        return [
            ("общение", 1),
            ("книги", 2),
            ("фильмы", 3),
            ("бизнес", 4),
            ("технологии", 5)
        ]
    elif language == 'de':
        return [
            ("kommunikation", 1),
            ("bücher", 2),
            ("filme", 3),
            ("geschäft", 4),
            ("technologie", 5)
        ]
    else:
        return [
            ("communication", 1),
            ("books", 2),
            ("movies", 3),
            ("business", 4),
            ("technology", 5)
        ]

@dp.message_handler(commands=['delete'])
async def delete_phrase_command(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_USERS:
        await message.reply("У вас нет разрешения на выполнение этой команды." if message.from_user.language_code == 'ru' else "You do not have permission to execute this command." if message.from_user.language_code == 'en' else "Sie haben keine Berechtigung, diesen Befehl auszuführen.")
        return

    args = message.get_args().split()
    if len(args) != 2:
        await message.reply("Пожалуйста, укажите как ID фразы, так и язык (ru/en/de). Пример: /delete 1 ru" if message.from_user.language_code == 'ru' else "Please provide both the phrase ID and the language (ru/en/de). Example: /delete 1 ru" if message.from_user.language_code == 'en' else "Bitte geben Sie sowohl die Phrasen-ID als auch die Sprache an (ru/en/de). Beispiel: /delete 1 ru")
        return

    phrase_id = args[0]
    language = args[1].strip().lower()
    if language not in ['ru', 'en', 'de']:
        await message.reply("Недопустимый язык. Укажите 'ru' для русского, 'en' для английского или 'de' для немецкого." if message.from_user.language_code == 'ru' else "Invalid language. Please specify 'ru' for Russian, 'en' for English or 'de' for German." if message.from_user.language_code == 'en' else "Ungültige Sprache. Bitte geben Sie 'ru' für Russisch, 'en' für Englisch oder 'de' für Deutsch an.")
        return

    try:
        phrase_id = int(phrase_id)
        delete_phrase(phrase_id, language)
        await message.reply(f"Фраза с ID {phrase_id} была удалена." if language == 'ru' else f"Phrase with ID {phrase_id} has been deleted." if language == 'en' else f"Phrase mit ID {phrase_id} wurde gelöscht.")
    except ValueError:
        await message.reply("Недопустимый ID фразы. Пожалуйста, укажите правильный числовой ID." if message.from_user.language_code == 'ru' else "Invalid phrase ID. Please provide a valid numeric ID." if message.from_user.language_code == 'en' else "Ungültige Phrasen-ID. Bitte geben Sie eine gültige numerische ID an.")
    except Exception as e:
        logging.error(f"Error deleting phrase: {e}")
        await message.reply("Произошла ошибка при удалении фразы." if message.from_user.language_code == 'ru' else "An error occurred while deleting the phrase." if message.from_user.language_code == 'en' else "Beim Löschen der Phrase ist ein Fehler aufgetreten.")

def delete_phrase(phrase_id, language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'DELETE FROM {table_name} WHERE id = ?', (phrase_id,))
    conn.commit()
    conn.close()
    logging.info(f"Deleted phrase with ID {phrase_id} from table {table_name}.")



def add_german_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved=0):
    phrase = phrase.lower()
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO german_phrases (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id))
    phrase_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Added German phrase: {phrase} with group_id: {group_id}, approved: {approved}, id: {phrase_id}, user_id: {user_id}")
    return phrase_id

def add_russian_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved=0):
    phrase = phrase.lower()
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO russian_phrases (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id))
    phrase_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Added Russian phrase: {phrase} with group_id: {group_id}, approved: {approved}, id: {phrase_id}, user_id: {user_id}")
    return phrase_id

def add_english_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved=0):
    phrase = phrase.lower()
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO english_phrases (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id))
    phrase_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Added English phrase: {phrase} with group_id: {group_id}, approved: {approved}, id: {phrase_id}, user_id: {user_id}")
    return phrase_id

def add_feedback(user_id, feedback):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (user_id, feedback, timestamp)
        VALUES (?, ?, ?)
    ''', (user_id, feedback, datetime.datetime.now().strftime('%Y-%m-%d')))
    conn.commit()
    conn.close()
    logging.info(f"Added feedback from user {user_id}: {feedback}")

def determine_language(phrase):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM russian_phrases WHERE phrase = ?', (phrase.lower(),))
    if cursor.fetchone()[0] > 0:
        logger.info(f"Phrase found in Russian phrases: {phrase}")
        return 'ru'
    
    cursor.execute('SELECT COUNT(*) FROM english_phrases WHERE phrase = ?', (phrase.lower(),))
    if cursor.fetchone()[0] > 0:
        logger.info(f"Phrase found in English phrases: {phrase}")
        return 'en'
    
    cursor.execute('SELECT COUNT(*) FROM german_phrases WHERE phrase = ?', (phrase.lower(),))
    if cursor.fetchone()[0] > 0:
        logger.info(f"Phrase found in German phrases: {phrase}")
        return 'de'
    
    conn.close()
    logger.warning(f"Phrase not found in any database: {phrase}")
    return None





def find_best_equivalent(phrase, target_lang, context_code=None, modality=None, intensity=None, meaning=None, context=None, threshold=0.2):
    try:
        src_lang = determine_language(phrase)
        if not src_lang:
            src_lang = detect_language(phrase)
        
        print(src_lang)
        interlingua_phrase = translate_to_interlingua(phrase, src_lang=src_lang, target_lang='en')
        phrase_embedding = get_embedding(interlingua_phrase)

        if target_lang == 'ru':
            table_name = 'german_phrases' if src_lang == 'de' else 'english_phrases'
            opposite_table_name = 'russian_phrases'
        elif target_lang == 'de':
            table_name = 'russian_phrases' if src_lang == 'ru' else 'english_phrases'
            opposite_table_name = 'german_phrases'
        else:
            table_name = 'russian_phrases' if src_lang == 'ru' else 'german_phrases'
            opposite_table_name = 'english_phrases'

        with sqlite3.connect('phrases.db') as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {table_name} WHERE phrase = ? AND approved = 1', (phrase.lower(),))
            row = cursor.fetchone()
            if not row:
                logging.warning(f"Phrase not found: {phrase}")
                return None

            phrase_id, group_id, phrase, modality, intensity, meaning, context, context_code_db, approved, user_id = row
            cursor.execute(f'SELECT * FROM {opposite_table_name} WHERE group_id = ? AND approved = 1', (group_id,))
            opposite_rows = cursor.fetchall()

        if not opposite_rows:
            logging.warning(f"No opposite language phrases found for group_id: {group_id}")
            return []

        weights = {
            'value': 0.3,
            'context_code_match': 0.2,
            'context_match': 0.2,
            'modality_match': 0.1,
            'intensity_match': 0.1,
            'meaning_match': 0.1
        }

        best_matches = []
        for opp_row in opposite_rows:
            opp_id, opp_group_id, opp_phrase, opp_modality, opp_intensity, opp_meaning, opp_context, opp_context_code, opp_approved, opp_user_id = opp_row
            opp_interlingua_phrase = translate_to_interlingua(opp_phrase, src_lang='en', target_lang='en')
            opp_embedding = get_embedding(opp_interlingua_phrase)

            similarity_embedding = calculate_similarity(phrase_embedding, opp_embedding)
            similarity_text = calculate_similarity(get_embedding(phrase), get_embedding(opp_phrase))
            similarity = max(similarity_embedding, similarity_text)

            similarity_score = similarity * weights['value']
            if context_code:
                similarity_score += (1 if context_code == opp_context_code else 0) * weights['context_code_match']
            if context:
                context_interlingua = translate_to_interlingua(context, src_lang='ru' if target_lang in ['en', 'de'] else 'en', target_lang='en')
                context_embedding = get_embedding(context_interlingua)
                opp_context_interlingua = translate_to_interlingua(opp_context, src_lang='en', target_lang='en')
                opp_context_embedding = get_embedding(opp_context_interlingua)
                context_similarity_embedding = calculate_similarity(context_embedding, opp_context_embedding)
                context_similarity_text = calculate_similarity(get_embedding(context), get_embedding(opp_context))
                context_similarity = max(context_similarity_embedding, context_similarity_text)
                similarity_score += context_similarity * weights['context_match']
            if modality:
                similarity_score += (1 if modality.lower() == opp_modality.lower() else 0) * weights['modality_match']
            if intensity is not None:
                similarity_score += (1 - abs(intensity - opp_intensity)) * weights['intensity_match']
            if meaning:
                meaning_interlingua = translate_to_interlingua(meaning, src_lang='ru' if target_lang in ['en', 'de'] else 'en', target_lang='en')
                meaning_embedding = get_embedding(meaning_interlingua)
                opp_meaning_interlingua = translate_to_interlingua(opp_meaning, src_lang='en', target_lang='en')
                opp_meaning_embedding = get_embedding(opp_meaning_interlingua)
                meaning_similarity_embedding = calculate_similarity(meaning_embedding, opp_meaning_embedding)
                meaning_similarity_text = calculate_similarity(get_embedding(meaning), get_embedding(opp_meaning))
                meaning_similarity = max(meaning_similarity_embedding, meaning_similarity_text)
                similarity_score += meaning_similarity * weights['meaning_match']

            similarity_score = min(similarity_score, 1.0)

            if similarity_score > threshold:
                best_matches.append((opp_phrase, similarity_score, opp_context, opp_meaning))

        best_matches.sort(key=lambda x: x[1], reverse=True)

        if best_matches:
            logging.info(f"Best matches for '{phrase}':")
            for match in best_matches:
                logging.info(f"'{match[0]}' with similarity score {match[1]} in context '{match[2]}'")
        else:
            logging.warning(f"No match found for '{phrase}'")

        return best_matches
    except Exception as e:
        logging.error(f"Error in find_best_equivalent: {e}")
        return []





def get_pending_phrases(language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table_name} WHERE approved = 0')
    rows = cursor.fetchall()
    conn.close()
    return rows

def approve_phrase(phrase_id, language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    try:
        phrase_id = int(phrase_id)
        conn = sqlite3.connect('phrases.db')
        cursor = conn.cursor()
        cursor.execute(f'UPDATE {table_name} SET approved = 1 WHERE id = ?', (phrase_id,))
        cursor.execute(f'SELECT user_id FROM {table_name} WHERE id = ?', (phrase_id,))
        user_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        logging.info(f"Phrase with ID {phrase_id} in table {table_name} has been approved.")
        if user_id:
            asyncio.create_task(bot.send_message(user_id, f"Your phrase with ID {phrase_id} has been approved and added to the database."))
    except ValueError:
        logging.error(f"Invalid phrase ID: {phrase_id}")
    except Exception as e:
        logging.error(f"Error approving phrase: {e}")


def phrase_exists(group_id, phrase, modality, intensity, meaning, context, context_code, language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT COUNT(*) FROM {table_name}
        WHERE group_id = ? AND phrase = ? AND modality = ? AND intensity = ? AND meaning = ? AND context = ? AND context_code = ?
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def get_last_group_id(language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT MAX(group_id) FROM {table_name}')
    last_group_id = cursor.fetchone()[0]
    conn.close()
    return last_group_id if last_group_id is not None else 0

def check_and_auto_add(group_id, phrase, modality, intensity, meaning, context, context_code, language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT COUNT(*) FROM {table_name}
        WHERE group_id = ? AND phrase = ? AND modality = ? AND intensity = ? AND meaning = ? AND context = ? AND context_code = ? AND approved = 0
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code))
    count = cursor.fetchone()[0]
    if count >= 5:
        cursor.execute(f'''
            UPDATE {table_name}
            SET approved = 1
            WHERE group_id = ? AND phrase = ? AND modality = ? AND intensity = ? AND meaning = ? AND context = ? AND context_code = ?
        ''', (group_id, phrase, modality, intensity, meaning, context, context_code))
        conn.commit()
        conn.close()
        logging.info(f"Phrase '{phrase}' auto-approved after {count} identical entries")
        return True
    conn.close()
    return False

# Dictionary examples for terms
modality_examples_ru = {
    'positive': "Пример положительной модальности: 'Отлично! Ты справился!'.",
    'negative': "Пример отрицательной модальности: 'Это было ужасно.'."
}

intensity_examples_ru = {
    '0': "Интенсивность 0: 'Это было неплохо.'",
    '1': "Интенсивность 1: 'Это было абсолютно потрясающе!'",
    '0.5': "Интенсивность 0.5: 'Это было довольно хорошо.'"
}

modality_examples_en = {
    'positive': "Example of positive modality: 'Great! You did it!'.",
    'negative': "Example of negative modality: 'That was awful.'."
}

intensity_examples_en = {
    '0': "Intensity 0: 'It was not bad.'",
    '1': "Intensity 1: 'It was absolutely amazing!'",
    '0.5': "Intensity 0.5: 'It was quite good.'."
}

context_examples_ru = {
    'общение': "Пример контекста 'общение': 'Мы обсуждали различные темы на встрече'.",
    'книги': "Пример контекста 'книги': 'Эта фраза часто встречается в литературе'.",
    'фильмы': "Пример контекста 'фильмы': 'Эта фраза используется в фильмах'.",
    'бизнес': "Пример контекста 'бизнес': 'Эта фраза часто используется в деловых переговорах'.",
    'технологии': "Пример контекста 'технологии': 'Эта фраза относится к техническим терминам'."
}

context_examples_en = {
    'communication': "Example context 'communication': 'We discussed various topics during the meeting'.",
    'books': "Example context 'books': 'This phrase is often found in literature'.",
    'movies': "Example context 'movies': 'This phrase is used in movies'.",
    'business': "Example context 'business': 'This phrase is often used in business negotiations'.",
    'technology': "Example context 'technology': 'This phrase is related to technical terms'."
}

def get_all_phrases(table_name):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT phrase FROM {table_name}')
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

@dp.message_handler(commands='list_russian_phrases')
async def list_russian_phrases(message: types.Message):
    phrases = get_all_phrases('russian_phrases')
    if phrases:
        await message.reply("Русские фразеологизмы:\n" + "\n".join(phrases))
    else:
        await message.reply("В базе данных нет русских фразеологизмов.")

@dp.message_handler(commands='list_german_phrases')
async def list_german_phrases(message: types.Message):
    phrases = get_all_phrases('german_phrases')
    if phrases:
        await message.reply("Немецкие фразеологизмы:\n" + "\n".join(phrases))
    else:
        await message.reply("В базе данных нет немецких фразеологизмов.")

@dp.message_handler(commands='list_english_phrases')
async def list_english_phrases(message: types.Message):
    phrases = get_all_phrases('english_phrases')
    if phrases:
        await message.reply("Английские фразеологизмы:\n" + "\n".join(phrases))
    else:
        await message.reply("В базе данных нет английских фразеологизмов.")


# Command and state handlers
@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    start_text = """
    Hello! I am a bot for entering phrases and their translations. Here are some things you can do:

    - Add a new phrase: /add
    Example: /add

    - Translate a phrase: /translate
    Example: /translate

    - Approve a phrase by ID: /approve <phrase_id> <language>
    Example: /approve 1 en

    - Approve all pending phrases in a language: /approve_all <language>
    Example: /approve_all en

    - Edit an existing phrase by ID: /edit
    Example: /edit

    - Delete a phrase by ID: /delete <phrase_id> <language>
    Example: /delete 1 en

    - Parse and add idioms to the database: /parse_idioms
    Example: /parse_idioms

    - Send feedback: /feedback
    Example: /feedback

    - Restart the bot: /restart
    Example: /restart

    - Exit the current state: /exit
    Example: /exit

    If you need more help, use the /help command.
    """
    await message.reply(start_text)


@dp.message_handler(commands='help')
async def cmd_help(message: types.Message):
    help_text = """
    Available commands:
    
    - Add a new phrase: /add
    Example: /add

    - Translate a phrase: /translate
    Example: /translate

    - Approve a phrase by ID: /approve <phrase_id> <language>
    Example: /approve 1 en

    - Approve all pending phrases in a language: /approve_all <language>
    Example: /approve_all en

    - Edit an existing phrase by ID: /edit
    Example: /edit

    - Delete a phrase by ID: /delete <phrase_id> <language>
    Example: /delete 1 en

    - Parse and add idioms to the database: /parse_idioms
    Example: /parse_idioms

    - Send feedback: /feedback
    Example: /feedback

    - List all Russian phrases: /list_russian_phrases
    Example: /list_russian_phrases

    - List all German phrases: /list_german_phrases
    Example: /list_german_phrases

    - List all English phrases: /list_english_phrases
    Example: /list_english_phrases

    - Restart the bot: /restart
    Example: /restart

    - Exit the current state: /exit
    Example: /exit
    """
    await message.reply(help_text)


@dp.message_handler(commands='exit', state='*')
async def cmd_exit(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.reply("You are not in any active state.")
        return

    await state.finish()
    await message.reply("You have exited the current process.")



@dp.message_handler(commands='add')
async def cmd_add(message: types.Message):
    await Form.language.set()
    await message.reply("Enter the language of the phrase (ru/en/de):")

@dp.message_handler(state=Form.language)
async def process_language(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return
    
    language = message.text.lower()
    if language not in ['ru', 'en', 'de']:
        await message.reply("Please enter a valid language (ru/en/de).")
        return

    async with state.proxy() as data:
        data['language'] = language
    
    last_group_id = get_last_group_id(language)
    await Form.group_id.set()
    await message.reply(f"Enter the group_id for the new phrase (last used group_id: {last_group_id}):")


@dp.message_handler(state=Form.group_id)
async def process_group_id(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return
    
    try:
        group_id = int(message.text)
        async with state.proxy() as data:
            data['group_id'] = group_id
        await Form.phrase.set()
        await message.reply("Enter the phrase:")
    except ValueError:
        await message.reply("Please enter a valid group_id.")



@dp.message_handler(state=Form.phrase)
async def process_phrase(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return
    
    async with state.proxy() as data:
        data['phrase'] = message.text.lower()
    
    if data['language'] == 'ru':
        await message.reply("Введите модальность (например, positive or negative). Примеры:\npositive: Пример положительной модальности: 'Отлично! Ты справился!'.\nnegative: Пример отрицательной модальности: 'Это было ужасно.'.")
    elif data['language'] == 'de':
        await message.reply("Geben Sie die Modalität ein (z. B. positive oder negative). Beispiele:\npositive: Beispiel für positive Modalität: 'Großartig! Du hast es geschafft!'.\nnegative: Beispiel für negative Modalität: 'Das war schrecklich.'.")
    else:
        await message.reply("Enter modality (e.g., positive or negative). Examples:\npositive: Example of positive modality: 'Great! You did it!'.\nnegative: Example of negative modality: 'That was awful.'.")
    
    await Form.modality.set()

@dp.message_handler(state=Form.modality)
async def process_modality(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['modality'] = message.text.lower()

    if data['language'] == 'ru':
        await message.reply("Введите Интенсивность (значение от 0 до 1). Примеры:\n0: Интенсивность 0: 'Это было неплохо.'\n1: Интенсивность 1: 'Это было абсолютно потрясающе!'\n0.5: Интенсивность 0.5: 'Это было довольно хорошо.'.")
    elif data['language'] == 'de':
        await message.reply("Geben Sie die Intensität ein (Wert von 0 bis 1). Beispiele:\n0: Intensität 0: 'Es war nicht schlecht.'\n1: Intensität 1: 'Es war absolut erstaunlich!'\n0.5: Intensität 0.5: 'Es war ziemlich gut.'.")
    else:
        await message.reply("Enter intensity (a numerical value from 0 to 1). Examples:\n0: Intensity 0: 'It was not bad.'\n1: Intensity 1: 'It was absolutely amazing!'\n0.5: Intensity 0.5: 'It was quite good.'.")
    
    await Form.intensity.set()


@dp.message_handler(state=Form.intensity)
async def process_intensity(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    try:
        intensity = float(message.text)
        async with state.proxy() as data:
            data['intensity'] = intensity
        await Form.meaning.set()

        if data['language'] == 'ru':
            await message.reply("Введите значение фразы")
        elif data['language'] == 'de':
            await message.reply("Geben Sie die Bedeutung der Phrase ein")
        else:
            await message.reply("Enter the meaning of the phrase (an explanation of what it means).")
    except ValueError:
        if data['language'] == 'ru':
            await message.reply("Пожалуйста, введите допустимое значение интенсивности.")
        elif data['language'] == 'de':
            await message.reply("Bitte geben Sie einen gültigen Intensitätswert ein.")
        else:
            await message.reply("Please enter a valid intensity value.")
        await Form.intensity.set()

@dp.message_handler(state=Form.meaning)
async def process_meaning(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['meaning'] = message.text.lower()

    if data['language'] == 'ru':
        await message.reply("Введите контекст (например, 'общение', 'книги', 'фильмы'). Примеры:\nобщение: Пример контекста 'общение': 'Мы обсуждали различные темы на встрече'.\nкниги: Пример контекста 'книги': 'Эта фраза часто встречается в литературе'.\nфильмы: Пример контекста 'фильмы': 'Эта фраза используется в фильмах'.")
    elif data['language'] == 'de':
        await message.reply("Geben Sie den Kontext ein (z. B. 'Kommunikation', 'Bücher', 'Filme'). Beispiele:\nKommunikation: Beispielkontext 'Kommunikation': 'Wir haben verschiedene Themen während des Treffens besprochen'.\nBücher: Beispielkontext 'Bücher': 'Diese Phrase findet sich oft in der Literatur'.\nFilme: Beispielkontext 'Filme': 'Diese Phrase wird in Filmen verwendet'.")
    else:
        await message.reply("Enter context (e.g., 'communication', 'books', 'movies'). Examples:\ncommunication: Example context 'communication': 'We discussed various topics during the meeting'.\nbooks: Example context 'books': 'This phrase is often found in literature'.\nmovies: Example context 'movies': 'This phrase is used in movies'.")
    
    await Form.context.set()

@dp.message_handler(state=Form.context)
async def process_context(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['context'] = message.text.lower()
        logger.info(f"Context: {data['context']}")

    context_examples = get_context_examples(data['language'])
    examples_text = "\n".join([f"{example[0]}: {example[1]}" for example in context_examples])

    if data['language'] == 'ru':
        await message.reply(f"Выберите код контекста из следующих примеров для '{data['context']}':\n{examples_text}")
    elif data['language'] == 'de':
        await message.reply(f"Wählen Sie einen Kontextcode aus den folgenden Beispielen für '{data['context']}':\n{examples_text}")
    else:
        await message.reply(f"Choose a context code from the following examples for '{data['context']}':\n{examples_text}")

    await Form.context_code.set()

async def finalize_addition(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        logger.info(f"Data in state before finalizing: {data}")
        phrase = data.get('phrase')
        context_code = data.get('context_code')
        modality = data.get('modality')
        intensity = data.get('intensity')
        meaning = data.get('meaning')
        context = data.get('context')
        group_id = data.get('group_id')
        language = data.get('language')
        approved = 0
        user_id = message.from_user.id

        logger.info(f"Final data - phrase: {phrase}, context_code: {context_code}, modality: {modality}, intensity: {intensity}, meaning: {meaning}, context: {context}, group_id: {group_id}, language: {language}, user_id: {user_id}")

        if not all(v is not None and v != '' for v in [phrase, modality, intensity, meaning, context, context_code, group_id, language, user_id]):
            await message.reply("Missing required data. Please try again.")
            await state.finish()
            return

        if phrase_exists(group_id, phrase, modality, intensity, meaning, context, context_code, language):
            if check_and_auto_add(group_id, phrase, modality, intensity, meaning, context, context_code, language):
                await message.reply("The phrase already exists and has been automatically approved.")
            else:
                await message.reply("The phrase already exists in the database and is awaiting approval.")
        else:
            if language == 'ru':
                add_russian_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved)
            elif language == 'en':
                add_english_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved)
            else:
                add_german_phrase(group_id, phrase, modality, intensity, meaning, context, context_code, user_id, approved)
                
            await bot.send_message(ADMIN_ID, f"New phrase awaiting approval from user_id {user_id}:\nGroup ID: {group_id}\nPhrase: {phrase}\nModality: {modality}\nIntensity: {intensity}\nMeaning: {meaning}\nContext: {context}\nContext Code: {context_code}\nLanguage: {language}")
            await message.reply("The phrase has been added and sent for approval.")

    await state.finish()


@dp.message_handler(state=Form.context_code)
async def process_context_code(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        context_code_input = message.text.lower()
        context_examples = get_context_examples(data['language'])
        context_dict = {str(code): context for context, code in context_examples}
        context_dict.update({context.lower(): code for context, code in context_examples})

        logger.info(f"Received context code input: {context_code_input}")
        logger.info(f"Context examples: {context_dict}")

        if context_code_input.isdigit():
            context_code = int(context_code_input)
            context_text = next((context for context, code in context_examples if code == context_code), None)
        elif context_code_input in context_dict:
            context_code = context_dict[context_code_input]
            context_text = next((context for context, code in context_examples if context.lower() == context_code_input), context_code_input)
        else:
            context_code = None
            context_text = None

        if context_code is None or context_text is None:
            examples_text = "\n".join([f"{code}: {context}" for context, code in context_examples])
            if data['language'] == 'ru':
                await message.reply(f"Неверный ввод контекста. Выберите код контекста из следующих примеров или введите 'no':\n{examples_text}")
            elif data['language'] == 'de':
                await message.reply(f"Ungültiger Kontexteingang. Wählen Sie einen Kontextcode aus den folgenden Beispielen oder geben Sie 'no' ein:\n{examples_text}")
            else:
                await message.reply(f"Invalid context input. Choose a context code from the following examples or enter 'no':\n{examples_text}")
            return

        data['context_code'] = context_code
        data['context'] = context_text

        logger.info(f"Context code set: {data['context_code']}, Context: {data['context']}")

        await state.update_data(context_code=context_code, context=context_text)
        await finalize_addition(message, state)




@dp.message_handler(commands='translate')
async def cmd_translate(message: types.Message):
    await Form.translation_phrase.set()
    await message.reply("Enter the phrase to translate:")

@dp.message_handler(state=Form.translation_phrase)
async def process_translation_phrase(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['phrase'] = message.text.lower()

    detected_lang = determine_language(message.text)
    if detected_lang is None:
        detected_lang = combined_language_detection(message.text)
    
    if detected_lang == 'conflict':
        await Form.waiting_for_language_confirmation.set()
        async with state.proxy() as data:
            data['detected_language'] = detected_lang
        await message.reply("The language of the phrase could not be determined accurately. Please specify the language (ru/en/de):")
    elif detected_lang == 'ru':
        async with state.proxy() as data:
            data['detected_language'] = detected_lang
        await Form.translation_language.set()
        await message.reply("На какой язык вы хотите перевести? (en/de)")
    elif detected_lang == 'en':
        async with state.proxy() as data:
            data['detected_language'] = detected_lang
        await Form.translation_language.set()
        await message.reply("To which language do you want to translate? (ru/de)")
    else:
        async with state.proxy() as data:
            data['detected_language'] = detected_lang
        await Form.translation_language.set()
        await message.reply("In welche Sprache möchten Sie übersetzen? (ru/en)")


@dp.message_handler(state=Form.waiting_for_language_confirmation)
async def process_language_confirmation(message: types.Message, state: FSMContext):
    language = message.text.lower()
    if language not in ['ru', 'en', 'de']:
        await message.reply("Please enter a valid language (ru/en/de).")
        return

    async with state.proxy() as data:
        data['detected_language'] = language
    
    await Form.translation_language.set()
    if language == 'ru':
        await message.reply("На какой язык вы хотите перевести? (en/de)")
    elif language == 'en':
        await message.reply("To which language do you want to translate? (ru/de)")
    else:
        await message.reply("In welche Sprache möchten Sie übersetzen? (ru/en)")



@dp.message_handler(state=Form.translation_language)
async def process_language_translation(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    language = message.text.lower()
    if language not in ['ru', 'en', 'de']:
        await message.reply("Please enter a valid language (ru/en/de).")
        return

    async with state.proxy() as data:
        data['target_language'] = language
        detected_lang = data.get('detected_language')

        if data['target_language'] == detected_lang:
            if detected_lang == 'ru':
                await message.reply("Невозможно перевести русскую фразу на русский язык. Пожалуйста, выберите другой язык перевода (en/de).")
            elif detected_lang == 'en':
                await message.reply("Translating an English phrase to English is not possible. Please choose another target language (ru/de).")
            else:
                await message.reply("Übersetzung einer deutschen Phrase ins Deutsche ist nicht möglich. Bitte wählen Sie eine andere Zielsprache (ru/en).")
            await Form.translation_language.set()
            return

    await Form.translation_process_choice.set()
    if detected_lang == 'ru':
        await message.reply("Хотите вручную определить параметры перевода (модальность, интенсивность, значение, контекст, код контекста)? Введите «yes» или «no».")
    elif detected_lang == 'en':
        await message.reply("Would you like to manually define parameters for the translation (modality, intensity, meaning, context, context_code)? Enter 'yes' or 'no'.")
    else:
        await message.reply("Möchten Sie die Parameter für die Übersetzung manuell festlegen (Modalität, Intensität, Bedeutung, Kontext, Kontextcode)? Geben Sie 'yes' ein, um zu bestätigen.")


@dp.message_handler(state=Form.translation_process_choice)
async def process_translation_choice(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    choice = message.text.lower()
    if choice not in ['yes', 'no']:
        await message.reply("Please enter 'yes' or 'no'.")
        return

    async with state.proxy() as data:
        data['manual_translation'] = choice == 'yes'
        source_language = data['detected_language']

    if choice == 'yes':
        await Form.manual_modality.set()
        if source_language == 'ru':
            await message.reply("Введите модальность (например, положительную-positive или отрицательную-negative или введите «skip», чтобы пропустить). Примеры:\nположительная модальность: 'Отлично! Ты сделал это!'.\noтрицательная модальность: 'Это было ужасно.'.")
        elif source_language == 'en':
            await message.reply("Enter modality (e.g., positive or negative, or type 'skip' to skip). Examples:\npositive: 'Great! You did it!'.\nnegative: 'That was awful.'.")
        else:
            await message.reply("Geben Sie die Modalität ein (z.B. positive oder negative, oder geben Sie 'skip' ein, um zu überspringen). Beispiele:\npositive: Beispiel für positive Modalität: 'Großartig! Du hast es geschafft!'.\nnegative: Beispiel für negative Modalität: 'Das war schrecklich.'.")
    else:
        await Form.manual_context.set()
        if source_language == 'ru':
            await message.reply("Введите контекст (например, 'общение', 'книги', 'фильмы' или введите 'skip', чтобы пропустить). Примеры:\nобщение: 'Во время встречи мы обсуждали различные темы'.\nкниги: 'Эта фраза часто встречается в литературе'.\nфильмы: 'Эта фраза используется в фильмах'.\nбизнес: 'Эта фраза часто используется в деловых переговорах'.\nтехнологии: 'Эта фраза относится к техническим терминам'.\nВы также можете ввести код контекста напрямую: 1 для общения, 2 для книг, 3 для фильмов, 4 для бизнеса, 5 для технологий.")
        elif source_language == 'en':
            await message.reply("Enter context (e.g., 'communication', 'books', 'movies', or type 'skip' to skip). Examples:\ncommunication: 'We discussed various topics during the meeting'.\nbooks: 'This phrase is often found in literature'.\nmovies: 'This phrase is used in movies'.\nbusiness: 'This phrase is often used in business negotiations'.\ntechnology: 'This phrase is related to technical terms'.\nYou can also enter the context code directly (1 for communication, 2 for books, 3 for movies, 4 for business, 5 for technology).")
        else:
            await message.reply("Geben Sie den Kontext ein (z.B. 'Kommunikation', 'Bücher', 'Filme' oder geben Sie 'skip' ein, um zu überspringen). Beispiele:\nKommunikation: 'Wir haben verschiedene Themen während des Treffens besprochen'.\nBücher: 'Diese Phrase findet sich oft in der Literatur'.\nFilme: 'Diese Phrase wird in Filmen verwendet'.\nGeschäft: 'Diese Phrase wird oft in Geschäftsverhandlungen verwendet'.\nTechnologie: 'Diese Phrase bezieht sich auf technische Begriffe'.\nSie können auch den Kontextcode direkt eingeben (1 für Kommunikation, 2 für Bücher, 3 für Filme, 4 für Geschäft, 5 für Technologie).")
            

@dp.message_handler(state=Form.manual_modality)
async def process_manual_modality(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['modality'] = message.text.lower() if message.text.lower() != 'skip' else None
        source_language = data['detected_language']

    await Form.manual_intensity.set()
    if source_language == 'ru':
        await message.reply("Введите интенсивность (значение от 0 до 1, или введите 'skip', чтобы пропустить). Примеры:\n0: 'Это было неплохо.'\n1: 'Это было абсолютно потрясающе!'\n0.5: 'Это было довольно хорошо.'.")
    elif source_language == 'en':
        await message.reply("Enter intensity (a numerical value from 0 to 1, or type 'skip' to skip). Examples:\n0: 'It was not bad.'\n1: 'It was absolutely amazing!'\n0.5: 'It was quite good.'.")
    else:
        await message.reply("Geben Sie die Intensität ein (ein numerischer Wert von 0 bis 1, oder geben Sie 'skip' ein, um zu überspringen). Beispiele:\n0: 'Es war nicht schlecht.'\n1: 'Es war absolut erstaunlich!'\n0.5: 'Es war ziemlich gut.'.")

@dp.message_handler(state=Form.manual_intensity)
async def process_manual_intensity(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    if message.text.lower() == 'skip':
        async with state.proxy() as data:
            data['intensity'] = None
            source_language = data['detected_language']

        await Form.manual_meaning.set()
        if source_language == 'ru':
            await message.reply("Введите значение фразы (или введите 'skip', чтобы пропустить).")
        elif source_language == 'en':
            await message.reply("Enter the meaning of the phrase (an explanation of what it means, or type 'skip' to skip).")
        else:
            await message.reply("Geben Sie die Bedeutung der Phrase ein (eine Erklärung, was sie bedeutet, oder geben Sie 'skip' ein, um zu überspringen).")
        return

    try:
        intensity = float(message.text)
        async with state.proxy() as data:
            data['intensity'] = intensity
            source_language = data['detected_language']

        await Form.manual_meaning.set()
        if source_language == 'ru':
            await message.reply("Введите значение фразы (или введите 'skip', чтобы пропустить).")
        elif source_language == 'en':
            await message.reply("Enter the meaning of the phrase (an explanation of what it means, or type 'skip' to skip).")
        else:
            await message.reply("Geben Sie die Bedeutung der Phrase ein (eine Erklärung, was sie bedeutet, oder geben Sie 'skip' ein, um zu überspringen).")
    except ValueError:
        async with state.proxy() as data:
            source_language = data['detected_language']
        if source_language == 'ru':
            await message.reply("Пожалуйста, введите допустимое значение интенсивности (или введите 'skip', чтобы пропустить).")
        elif source_language == 'en':
            await message.reply("Please enter a valid intensity value (or type 'skip' to skip).")
        else:
            await message.reply("Bitte geben Sie einen gültigen Intensitätswert ein (oder geben Sie 'skip' ein, um zu überspringen).")
        await Form.manual_intensity.set()



@dp.message_handler(state=Form.manual_meaning)
async def process_manual_meaning(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    async with state.proxy() as data:
        data['meaning'] = message.text.lower() if message.text.lower() != 'skip' else None
        source_language = data['detected_language']

    await Form.manual_context.set()
    if source_language == 'ru':
        await message.reply("Введите контекст (например, 'общение', 'книги', 'фильмы' или введите 'skip', чтобы пропустить). Примеры:\nобщение: 'Во время встречи мы обсуждали различные темы'.\nкниги: 'Эта фраза часто встречается в литературе'.\nфильмы: 'Эта фраза используется в фильмах'.\nбизнес: 'Эта фраза часто используется в деловых переговорах'.\nтехнологии: 'Эта фраза относится к техническим терминам'.")
    elif source_language == 'en':
        await message.reply("Enter context (e.g., 'communication', 'books', 'movies', or type 'skip' to skip). Examples:\ncommunication: 'We discussed various topics during the meeting'.\nbooks: 'This phrase is often found in literature'.\nmovies: 'This phrase is used in movies'.\nbusiness: 'This phrase is often used in business negotiations'.\ntechnology: 'This phrase is related to technical terms'.")
    else:
        await message.reply("Geben Sie den Kontext ein (z.B. 'Kommunikation', 'Bücher', 'Filme' oder geben Sie 'skip' ein, um zu überspringen). Beispiele:\nKommunikation: 'Wir haben verschiedene Themen während des Treffens besprochen'.\nBücher: 'Diese Phrase findet sich oft in der Literatur'.\nFilme: 'Diese Phrase wird in Filmen verwendet'.\nGeschäft: 'Diese Phrase wird oft in Geschäftsverhandlungen verwendet'.\nTechnologie: 'Diese Phrase bezieht sich auf technische Begriffe'.")



@dp.message_handler(state=Form.manual_context)
async def process_manual_context(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    if message.text.lower() == 'skip':
        async with state.proxy() as data:
            data['context'] = None
            data['context_code'] = None
        await finalize_translation(message, state)
        return

    async with state.proxy() as data:
        context_input = message.text.lower()
        source_language = data['detected_language']
        context_examples = get_context_examples(source_language)
        context_dict = {str(code): context for context, code in context_examples}
        context_dict.update({context.lower(): code for context, code in context_examples})

        logger.info(f"Received context input: {context_input}")
        logger.info(f"Context examples: {context_dict}")

        if context_input.isdigit():
            context_code = int(context_input)
            context_text = next((context for context, code in context_examples if code == context_code), None)
        elif context_input in context_dict:
            context_code = context_dict[context_input]
            context_text = next((context for context, code in context_examples if context.lower() == context_input), context_input)
        else:
            context_code = None
            context_text = None

        if context_code is None or context_text is None:
            examples_text = "\n".join([f"{code}: {context}" for context, code in context_examples])
            if source_language == 'ru':
                await message.reply(f"Неверный ввод контекста. Выберите код контекста из следующих примеров или введите 'skip', чтобы пропустить:\n{examples_text}")
            elif source_language == 'en':
                await message.reply(f"Invalid context input. Choose a context code from the following examples or enter 'skip' to skip:\n{examples_text}")
            else:
                await message.reply(f"Ungültige Kontexteingabe. Wählen Sie einen Kontextcode aus den folgenden Beispielen oder geben Sie 'skip' ein, um zu überspringen:\n{examples_text}")
            return

        data['context_code'] = context_code
        data['context'] = context_text

        logger.info(f"Context code set: {data['context_code']}, Context: {data['context']}")

        await state.update_data(context_code=context_code, context=context_text)
        await finalize_translation(message, state)


def translate_text(text, src_lang, tgt_lang):
    if src_lang == 'ru' and tgt_lang == 'de':
        intermediate_lang = 'en'
        translated_text = translate_text(text, src_lang, intermediate_lang)
        return translate_text(translated_text, intermediate_lang, tgt_lang)
    elif src_lang == 'de' and tgt_lang == 'ru':
        intermediate_lang = 'en'
        translated_text = translate_text(text, src_lang, intermediate_lang)
        return translate_text(translated_text, intermediate_lang, tgt_lang)
    else:
        if (src_lang, tgt_lang) not in translation_models:
            return "Unsupported language pair"
        
        model = translation_models[(src_lang, tgt_lang)]
        tokenizer = translation_tokenizers[(src_lang, tgt_lang)]
        
        inputs = tokenizer.encode(text, return_tensors='pt')
        translated = model.generate(inputs, max_length=512)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text





async def finalize_translation(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        phrase = data['phrase']
        target_language = data['target_language']
        context_code = data.get('context_code')
        modality = data.get('modality')
        intensity = data.get('intensity')
        meaning = data.get('meaning')
        context = data.get('context')

        logger.info(f"Final data - phrase: {phrase}, target_language: {target_language}, context_code: {context_code}, modality: {modality}, intensity: {intensity}, meaning: {meaning}, context: {context}")

        src_lang = data.get('detected_language')
        if not src_lang:
            src_lang = determine_language(phrase)
            if not src_lang:
                src_lang = detect_language(phrase)
            data['detected_language'] = src_lang

        translations = find_best_equivalent(phrase, target_language, context_code=context_code, modality=modality, intensity=intensity, meaning=meaning, context=context)

        if translations:
            if target_language == 'ru':
                translation_text = "\n".join([f"{phrase} (контекст: {context}, оценка: {score}, значение: {translate_to_interlingua(meaning, src_lang='en', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                await message.reply(f"Переводы:\n{translation_text}")
            elif target_language == 'de':
                translation_text = "\n".join([f"{phrase} (Kontext: {context}, Bewertung: {score}, Bedeutung: {translate_to_interlingua(meaning, src_lang='en', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                await message.reply(f"Übersetzungen:\n{translation_text}")
            else:
                translation_text = "\n".join([f"{phrase} (context: {context}, score: {score}, meaning: {translate_to_interlingua(meaning, src_lang='ru', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                await message.reply(f"Translations:\n{translation_text}")
        else:
            if target_language == 'ru':
                await message.reply("Подходящий перевод не найден.")
            elif target_language == 'de':
                await message.reply("Keine passende Übersetzung gefunden.")
            else:
                await message.reply("No suitable translation found.")
                
            translations = find_best_equivalent(phrase, target_language)
            if translations:
                if target_language == 'ru':
                    translation_text = "\n".join([f"{phrase} (контекст: {context}, оценка: {score}, значение: {translate_to_interlingua(meaning, src_lang='en', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                    await message.reply(f"Все переводы:\n{translation_text}")
                elif target_language == 'de':
                    translation_text = "\n".join([f"{phrase} (Kontext: {context}, Bewertung: {score}, Bedeutung: {translate_to_interlingua(meaning, src_lang='en', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                    await message.reply(f"Alle Übersetzungen:\n{translation_text}")
                else:
                    translation_text = "\n".join([f"{phrase} (context: {context}, score: {score}, meaning: {translate_to_interlingua(meaning, src_lang='ru', target_lang=target_language)})" for phrase, score, context, meaning in translations])
                    await message.reply(f"All translations:\n{translation_text}")
            else:
                if target_language == 'ru':
                    await message.reply("Доступные переводы для этой фразы отсутствуют.")
                elif target_language == 'de':
                    await message.reply("Keine verfügbaren Übersetzungen für diese Phrase.")
                else:
                    await message.reply("No available translations for this phrase.")

        ai_translation = translate_text(phrase, src_lang=src_lang, tgt_lang=target_language)
        additional_translations = [ai_translation]

        if additional_translations:
            if target_language == 'ru':
                await message.reply(f"Дополнительные переводы:\n" + "\n".join(additional_translations))
            elif target_language == 'de':
                await message.reply(f"Zusätzliche Übersetzungen:\n" + "\n".join(additional_translations))
            else:
                await message.reply(f"Additional translations:\n" + "\n".join(additional_translations))
        else:
            if target_language == 'ru':
                await message.reply("Дополнительные переводы не найдены.")
            elif target_language == 'de':
                await message.reply("Keine zusätzlichen Übersetzungen gefunden.")
            else:
                await message.reply("No additional translations found.")

    await state.finish()










@dp.message_handler(commands='approve')
async def approve_phrase_command(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_USERS:
        await message.reply("У вас нет разрешения на выполнение этой команды." if message.text.lower() == 'ru' else "You do not have permission to execute this command." if message.text.lower() == 'en' else "Sie haben keine Berechtigung, diesen Befehl auszuführen.")
        return

    try:
        args = message.get_args().split()
        if len(args) != 2:
            await message.reply("Пожалуйста, укажите как ID фразы, так и язык (ru/en/de). Пример: /approve 1 ru" if message.text.lower() == 'ru' else "Please provide both the phrase ID and the language (ru/en/de). Example: /approve 1 ru" if message.text.lower() == 'en' else "Bitte geben Sie sowohl die Phrasen-ID als auch die Sprache an (ru/en/de). Beispiel: /approve 1 ru")
            return
        phrase_id = args[0]
        language = args[1].strip().lower()
        if language not in ['ru', 'en', 'de']:
            await message.reply("Недопустимый язык. Укажите 'ru' для русского, 'en' для английского или 'de' для немецкого." if language == 'ru' else "Invalid language. Please specify 'ru' for Russian, 'en' for English or 'de' for German." if language == 'en' else "Ungültige Sprache. Bitte geben Sie 'ru' für Russisch, 'en' für Englisch oder 'de' für Deutsch an.")
            return
        approve_phrase(phrase_id, language)
        await message.reply(f"Фраза с ID {phrase_id} на языке {language} была одобрена." if language == 'ru' else f"Phrase with ID {phrase_id} in language {language} has been approved." if language == 'en' else f"Phrase mit ID {phrase_id} in Sprache {language} wurde genehmigt.")
    except Exception as e:
        logging.error(f"Error approving phrase: {e}")
        await message.reply("Произошла ошибка при одобрении фразы." if message.text.lower() == 'ru' else "An error occurred while approving the phrase." if message.text.lower() == 'en' else "Beim Genehmigen der Phrase ist ein Fehler aufgetreten.")

def approve_all_phrases(language):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'UPDATE {table_name} SET approved = 1 WHERE approved = 0')
    conn.commit()
    conn.close()


@dp.message_handler(commands='approve_all')
async def approve_all_phrases_command(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_USERS:
        await message.reply("У вас нет разрешения на выполнение этой команды." if message.text.lower() == 'ru' else "You do not have permission to execute this command." if message.text.lower() == 'en' else "Sie haben keine Berechtigung, diesen Befehl auszuführen.")
        return
    
    try:
        language = message.get_args().strip().lower()
        if language not in ['ru', 'en', 'de']:
            await message.reply("Недопустимый язык. Укажите 'ru' для русского, 'en' для английского или 'de' для немецкого." if language == 'ru' else "Invalid language. Please specify 'ru' for Russian, 'en' for English or 'de' for German." if language == 'en' else "Ungültige Sprache. Bitte geben Sie 'ru' für Russisch, 'en' für Englisch oder 'de' für Deutsch an.")
            return
        approve_all_phrases(language)
        await message.reply("Все фразы одобрены." if language == 'ru' else "All phrases approved." if language == 'en' else "Alle Phrasen genehmigt.")
    except Exception as e:
        logging.error(f"Error approving all phrases: {e}")
        await message.reply("Произошла ошибка при одобрении всех фраз." if language == 'ru' else "An error occurred while approving all phrases." if language == 'en' else "Beim Genehmigen aller Phrasen ist ein Fehler aufgetreten.")

@dp.message_handler(commands=['edit'])
async def edit_phrase(message: types.Message):
    await FormEdit.phrase_id.set()
    await message.reply("Введите ID фразы для редактирования:" if message.from_user.language_code == 'ru' else "Enter the phrase ID to edit:" if message.from_user.language_code == 'en' else "Geben Sie die ID der Phrase ein, die Sie bearbeiten möchten:")

class FormEdit(StatesGroup):
    phrase_id = State()
    language = State()
    phrase = State()
    modality = State()
    intensity = State()
    meaning = State()
    context = State()
    context_code = State()

class FormFeedback(StatesGroup):
    feedback = State()
    feedback_user_id = State()

@dp.message_handler(state=FormEdit.phrase_id)
async def process_phrase_id(message: types.Message, state: FSMContext):
    phrase_id = message.text.strip()
    if not phrase_id.isdigit():
        await message.reply("Введите допустимый числовой ID фразы." if message.from_user.language_code == 'ru' else "Please enter a valid numeric phrase ID." if message.from_user.language_code == 'en' else "Bitte geben Sie eine gültige numerische Phrasen-ID ein.")
        return
    
    phrase_id = int(phrase_id)
    async with state.proxy() as data:
        data['phrase_id'] = phrase_id
    
    await FormEdit.language.set()
    await message.reply("Введите язык фразы (ru/en/de):" if message.from_user.language_code == 'ru' else "Enter the language of the phrase (ru/en/de):" if message.from_user.language_code == 'en' else "Geben Sie die Sprache der Phrase ein (ru/en/de):")


@dp.message_handler(state=FormEdit.language)
async def process_language_edit(message: types.Message, state: FSMContext):
    language = message.text.strip().lower()
    if language not in ['ru', 'en', 'de']:
        await message.reply("Пожалуйста, введите допустимый язык (ru/en/de)." if message.from_user.language_code == 'ru' else "Please enter a valid language (ru/en/de)." if message.from_user.language_code == 'en' else "Bitte geben Sie eine gültige Sprache ein (ru/en/de).")
        return
    
    async with state.proxy() as data:
        data['language'] = language
        phrase = get_phrase_by_id(data['phrase_id'], language)
        if not phrase:
            await message.reply(f"Фраза с ID {data['phrase_id']} не найдена." if language == 'ru' else f"Phrase with ID {data['phrase_id']} not found." if language == 'en' else f"Phrase mit ID {data['phrase_id']} nicht gefunden.")
            await state.finish()
            return
        data['group_id'] = phrase[1]
        data['phrase'] = phrase[2]
        data['modality'] = phrase[3]
        data['intensity'] = phrase[4]
        data['meaning'] = phrase[5]
        data['context'] = phrase[6]
        data['context_code'] = phrase[7]
        data['approved'] = phrase[8]
    
    await FormEdit.phrase.set()
    await message.reply(f"Текущая фраза: {data['phrase']}\nВведите новую фразу (или введите 'skip', чтобы оставить текущее значение):" if language == 'ru' else f"Current phrase: {data['phrase']}\nEnter the new phrase (or type 'skip' to keep the current value):" if language == 'en' else f"Aktuelle Phrase: {data['phrase']}\nGeben Sie die neue Phrase ein (oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten):")

@dp.message_handler(state=FormEdit.phrase)
async def process_new_phrase(message: types.Message, state: FSMContext):
    new_phrase = message.text.strip()
    async with state.proxy() as data:
        if new_phrase.lower() != 'skip':
            data['phrase'] = new_phrase.lower()
    
    await FormEdit.modality.set()
    await message.reply(f"Текущая модальность: {data['modality']}\nВведите новую модальность (или введите 'skip', чтобы оставить текущее значение):" if data['language'] == 'ru' else f"Current modality: {data['modality']}\nEnter the new modality (or type 'skip' to keep the current value):" if data['language'] == 'en' else f"Aktuelle Modalität: {data['modality']}\nGeben Sie die neue Modalität ein (oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten):")

@dp.message_handler(state=FormEdit.modality)
async def process_new_modality(message: types.Message, state: FSMContext):
    new_modality = message.text.strip()
    async with state.proxy() as data:
        if new_modality.lower() != 'skip':
            data['modality'] = new_modality.lower()
    
    await FormEdit.intensity.set()
    await message.reply(
        f"Текущая интенсивность: {data['intensity']}\nВведите новую интенсивность (значение от 0 до 1, или введите 'skip', чтобы оставить текущее значение):" if data['language'] == 'ru' 
        else f"Current intensity: {data['intensity']}\nEnter the new intensity (a value from 0 to 1, or type 'skip' to keep the current value):" 
        if data['language'] == 'en' 
        else f"Aktuelle Intensität: {data['intensity']}\nGeben Sie die neue Intensität ein (ein Wert von 0 bis 1, oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten):"
    )
@dp.message_handler(state=FormEdit.intensity)
async def process_new_intensity(message: types.Message, state: FSMContext):
    new_intensity = message.text.strip()
    async with state.proxy() as data:
        if new_intensity.lower() != 'skip':
            try:
                data['intensity'] = float(new_intensity)
            except ValueError:
                await message.reply(
                    "Введите допустимое значение интенсивности (число от 0 до 1)." if data['language'] == 'ru' 
                    else "Please enter a valid intensity value (a number from 0 to 1)." 
                    if data['language'] == 'en' 
                    else "Bitte geben Sie einen gültigen Intensitätswert ein (eine Zahl von 0 bis 1)."
                )
                return
    
    await FormEdit.meaning.set()
    await message.reply(
        f"Текущее значение: {data['meaning']}\nВведите новое значение (или введите 'skip', чтобы оставить текущее значение):" if data['language'] == 'ru' 
        else f"Current meaning: {data['meaning']}\nEnter the new meaning (or type 'skip' to keep the current value):" 
        if data['language'] == 'en' 
        else f"Aktuelle Bedeutung: {data['meaning']}\nGeben Sie die neue Bedeutung ein (oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten):"
    )

@dp.message_handler(state=FormEdit.meaning)
async def process_new_meaning(message: types.Message, state: FSMContext):
    new_meaning = message.text.strip()
    async with state.proxy() as data:
        if new_meaning.lower() != 'skip':
            data['meaning'] = new_meaning.lower()
    
    await FormEdit.context.set()
    if data['language'] == 'ru':
        await message.reply(f"Текущий контекст: {data['context']}\nВведите новый контекст (или введите 'skip', чтобы оставить текущее значение). Примеры:\n" +
                            "\n".join([f"{v}: {k}" for k, v in get_context_examples(data['language'])]))
    elif data['language'] == 'de':
        await message.reply(f"Aktueller Kontext: {data['context']}\nGeben Sie den neuen Kontext ein (oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten). Beispiele:\n" +
                            "\n".join([f"{v}: {k}" for k, v in get_context_examples(data['language'])]))
    else:
        await message.reply(f"Current context: {data['context']}\nEnter the new context (or type 'skip' to keep the current value). Examples:\n" +
                            "\n".join([f"{v}: {k}" for k, v in get_context_examples(data['language'])]))


@dp.message_handler(state=FormEdit.context)
async def process_new_context(message: types.Message, state: FSMContext):
    new_context = message.text.strip()
    async with state.proxy() as data:
        if new_context.lower() != 'skip':
            data['context'] = new_context.lower()
    
    context_examples = get_context_examples(data['language'])
    examples_text = "\n".join([f"{code}: {context}" for context, code in context_examples])
    
    if data['language'] == 'ru':
        await message.reply(f"Текущий код контекста: {data['context_code']}\nВыберите код контекста из следующих примеров для '{data['context']}':\n{examples_text}\n(или введите 'skip', чтобы оставить текущее значение).")
    elif data['language'] == 'de':
        await message.reply(f"Aktueller Kontextcode: {data['context_code']}\nWählen Sie einen Kontextcode aus den folgenden Beispielen für '{data['context']}':\n{examples_text}\n(oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten).")
    else:
        await message.reply(f"Current context code: {data['context_code']}\nChoose a context code from the following examples for '{data['context']}':\n{examples_text}\n(or type 'skip' to keep the current value).")
    
    await FormEdit.context_code.set()


@dp.message_handler(state=FormEdit.context_code)
async def process_new_context_code(message: types.Message, state: FSMContext):
    context_code_input = message.text.lower()
    async with state.proxy() as data:
        if context_code_input == 'skip':
            await finalize_edit(message, state)
            return

        context_examples = get_context_examples(data['language'])
        context_dict = {str(code): context for context, code in context_examples}
        context_dict.update({context.lower(): code for context, code in context_examples})

        if context_code_input.isdigit() and int(context_code_input) in context_dict:
            context_code = int(context_code_input)
            context_text = next(context for context, code in context_examples if code == context_code)
        elif context_code_input in context_dict:
            context_code = context_dict[context_code_input]
            context_text = next(context for context, code in context_examples if context.lower() == context_code_input)
        else:
            context_code = None

        if context_code is None:
            examples_text = "\n".join([f"{code}: {context}" for context, code in context_examples])
            if data['language'] == 'ru':
                await message.reply(f"Недопустимый ввод контекста. Выберите код контекста из следующих примеров или введите 'skip', чтобы оставить текущее значение:\n{examples_text}")
            elif data['language'] == 'de':
                await message.reply(f"Ungültige Kontexteingabe. Wählen Sie einen Kontextcode aus den folgenden Beispielen oder geben Sie 'skip' ein, um den aktuellen Wert beizubehalten:\n{examples_text}")
            else:
                await message.reply(f"Invalid context input. Choose a context code from the following examples or enter 'skip' to keep the current value:\n{examples_text}")
            return

        data['context_code'] = context_code
        data['context'] = context_text

        await finalize_edit(message, state)

async def finalize_edit(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        phrase_id = data.get('phrase_id')
        group_id = data.get('group_id')
        phrase = data.get('phrase')
        context_code = data.get('context_code')
        modality = data.get('modality')
        intensity = data.get('intensity')
        meaning = data.get('meaning')
        context = data.get('context')
        approved = data.get('approved')
        language = data.get('language')
        user_id = message.from_user.id

        if not all([phrase, modality, intensity, meaning, context, context_code is not None, group_id, language]):
            await message.reply("Данные отсутствуют. Пожалуйста, попробуйте еще раз." if language == 'ru' else "Missing required data. Please try again." if language == 'en' else "Daten fehlen. Bitte versuchen Sie es erneut.")
            await state.finish()
            return

        update_phrase(phrase_id, group_id, phrase, modality, intensity, meaning, context, context_code, approved, language, user_id)
        await message.reply(f"Фраза с ID {phrase_id} была обновлена." if language == 'ru' else f"Phrase with ID {phrase_id} has been updated." if language == 'en' else f"Phrase mit ID {phrase_id} wurde aktualisiert.")
                
    await state.finish()


def update_phrase(phrase_id, group_id, phrase, modality, intensity, meaning, context, context_code, approved, language, user_id):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        UPDATE {table_name}
        SET group_id = ?, phrase = ?, modality = ?, intensity = ?, meaning = ?, context = ?, context_code = ?, approved = ?, user_id = ?
        WHERE id = ?
    ''', (group_id, phrase, modality, intensity, meaning, context, context_code, approved, user_id, phrase_id))
    conn.commit()
    conn.close()


def get_phrase_by_id(phrase_id: int, language: str):
    table_name = 'russian_phrases' if language == 'ru' else 'english_phrases' if language == 'en' else 'german_phrases'
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table_name} WHERE id = ?', (phrase_id,))
    phrase_row = cursor.fetchone()
    conn.close()
    return phrase_row

def approve_phrase_in_db(phrase_id: int):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE russian_phrases SET approved = 1 WHERE id = ?', (phrase_id,))
    conn.commit()
    conn.close()

def create_idioms_table():
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS idioms (
        id INTEGER PRIMARY KEY,
        idiom TEXT NOT NULL UNIQUE,
        meaning TEXT,
        origin TEXT
    )
    ''')
    conn.commit()
    conn.close()
    logger.debug("Idioms table created or already exists.")

def idiom_exists(idiom):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM idioms WHERE idiom = ?', (idiom,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def parse_idioms(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        tree = html.fromstring(response.content)

        idioms = []
        idioms_list_items = tree.xpath("//div[@class='mw-content-ltr mw-parser-output']//tr")
        logging.debug(f'Found {len(idioms_list_items)} items in the list')

        for item in idioms_list_items:
            idiom_text = item.xpath('.//td[1]//text()')
            meaning = item.xpath('.//td[2]//text()')
            origin = item.xpath('.//td[3]//text()')

            if idiom_text:
                idiom_text = " ".join(idiom_text).strip()
                meaning = " ".join(meaning).strip() if meaning else None
                origin = " ".join(origin).strip() if origin else None
                
                logging.debug(f'Idiom: {idiom_text}, Meaning: {meaning}, Origin: {origin}')
                
                idioms.append((idiom_text, meaning, origin))
        
        logging.debug(f'Total parsed idioms: {len(idioms)}')
        return idioms
    except requests.RequestException as e:
        logging.error(f"Error requesting page: {e}")
        return []

def add_idioms_to_db(idioms):
    if not idioms:
        logger.debug("No idioms to add to the database.")
        return 0

    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    
    added_count = 0
    for idiom, meaning, origin in idioms:
        if not idiom_exists(idiom):
            try:
                logging.debug(f'Adding to database: {idiom, meaning, origin}')
                cursor.execute('''
                    INSERT INTO idioms (idiom, meaning, origin)
                    VALUES (?, ?, ?)
                ''', (idiom, meaning, origin))
                added_count += 1
            except sqlite3.IntegrityError as e:
                logger.error(f'Error adding {idiom}: {e}')
        else:
            logger.debug(f'Idiom already exists in the database: {idiom}')
    
    conn.commit()
    conn.close()
    logger.debug(f'Added {added_count} idioms to the database.')
    return added_count

@dp.message_handler(commands=['parse_idioms'])
async def parse_idioms_command(message: types.Message):
    url = 'https://ru.wiktionary.org/wiki/Приложение:Список_фразеологизмов_русского_языка'
    idioms = parse_idioms(url)
    added_count = add_idioms_to_db(idioms)
    logging.debug(f"Added {added_count} idioms to the database.")
    await message.reply(f"Added {added_count} idioms to the database.")



@dp.message_handler(commands=['feedback'])
async def cmd_feedback(message: types.Message):
    if message.from_user.id in ADMIN_USERS:
        await FormFeedback.feedback_user_id.set()
        await message.reply("Please enter the user_id you want to send feedback to:")
    else:
        await FormFeedback.feedback.set()
        await message.reply("Please enter your feedback:")

@dp.message_handler(state=FormFeedback.feedback_user_id)
async def process_feedback_user_id(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return
    user_id = message.text.strip()
    if not user_id.isdigit():
        await message.reply("Please enter a valid user ID.")
        return
    
    async with state.proxy() as data:
        data['feedback_user_id'] = int(user_id)
    await FormFeedback.feedback.set()
    await message.reply("Please enter your feedback:")

@dp.message_handler(state=FormFeedback.feedback)
async def process_feedback(message: types.Message, state: FSMContext):
    if message.text.lower() == '/exit':
        await cmd_exit(message, state)
        return

    user_id = message.from_user.id
    username = message.from_user.username
    feedback_text = message.text.strip()

    async with state.proxy() as data:
        feedback_user_id = data.get('feedback_user_id')
    
    if feedback_user_id:
        try:
            await bot.send_message(feedback_user_id, f"Feedback from admin: {feedback_text}")
            await message.reply("Feedback has been sent to the user.")
        except Exception as e:
            await message.reply(f"Failed to send feedback to user: {e}")
    else:
        if has_given_feedback_today(user_id):
            await message.reply("You have already provided feedback today. Please try again tomorrow.")
            await state.finish()
            return

        save_feedback(user_id, username, feedback_text)
        await bot.send_message(ADMIN_ID, f"New feedback received from @{username} (user_id: {user_id}):\n{feedback_text}")
        await message.reply("Thank you for your feedback!")

    await state.finish()


def has_given_feedback_today(user_id):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) FROM feedback
        WHERE user_id = ? AND date(timestamp) = date('now')
    ''', (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def save_feedback(user_id, username, feedback):
    conn = sqlite3.connect('phrases.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (user_id, username, feedback, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    ''', (user_id, username, feedback))
    conn.commit()
    conn.close()


# Определение состояния для команды рестарт
class FormRestart(StatesGroup):
    confirmation = State()

# Функция для перезапуска бота
@dp.message_handler(commands='restart')
async def cmd_restart(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_USERS:
        await message.reply("У вас нет разрешения на выполнение этой команды." if message.from_user.language_code == 'ru' else "You do not have permission to execute this command." if message.from_user.language_code == 'en' else "Sie haben keine Berechtigung, diesen Befehl auszuführen.")
        return

    await FormRestart.confirmation.set()
    await message.reply("Вы уверены, что хотите перезапустить бота? Введите 'yes' для подтверждения." if message.from_user.language_code == 'ru' else "Are you sure you want to restart the bot? Type 'yes' to confirm." if message.from_user.language_code == 'en' else "Sind Sie sicher, dass Sie den Bot neu starten möchten? Geben Sie 'yes' ein, um zu bestätigen.")

@dp.message_handler(state=FormRestart.confirmation)
async def process_restart_confirmation(message: types.Message, state: FSMContext):
    if message.text.lower() == 'yes':
        await message.reply("Перезапуск бота..." if message.from_user.language_code == 'ru' else "Restarting bot..." if message.from_user.language_code == 'en' else "Bot wird neu gestartet...")
        await state.finish()
        python = sys.executable
        os.execl(python, python, * sys.argv)
    else:
        await message.reply("Рестарт отменен." if message.from_user.language_code == 'ru' else "Restart canceled." if message.from_user.language_code == 'en' else "Neustart abgebrochen.")
        await state.finish()

# Регулярное обновление информации каждые 10 секунд
async def periodic_update():
    while True:
        try:
            print("1111111111111111111")
            await asyncio.sleep(20)
        except Exception as e:
            logger.error(f"Error during periodic update: {e}")
            print("Non-critical error received")
    
if __name__ == '__main__':
    create_tables()
    create_idioms_table()
    create_feedback_table()
    #loop = asyncio.get_event_loop()
    #loop.create_task(periodic_update())
    executor.start_polling(dp, skip_updates=True)



