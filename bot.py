"""
Author: Harshavardhan
"""
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
import telegram
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
import logging
import cv2
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.backend import set_session
from classification_models.keras import Classifiers
import keras
import ctypes
import webbrowser as wb
import numpy as np
import pandas as pd
import tensorflow as tf
import absl.logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
import smtplib
import time
import joblib
import json
import requests

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

# GLOBAL VARIABLES
BOT_TOKEN = '938059809:AAGbZ0bub6QM-LoRaesWR585TEOUxw1w_YY'
WEATHER_KEY = '694a0650817a49655d2b7934ecf1aa03'
BOT_CHAT_ID = None  # Bot chat ID is no longer needed for starting the bot. Its dynamically picked
# BOT_TOKEN = '918870404:AAGylpk_4euuNUUMidmVGI3csLBslhxJiwE' # shyam token


# MODEL RELATED
MODEL = None
PREPROCESS_INPUT = None
GRAPH = None
SESS = None
INIT = None
MODEL_TO_FETCH = 'nasnetlarge'

FASTAI_MODEL = None

# EMAIL RELATED
RECIPIENT, ASK_SUBJECT, ASK_MESSAGE, ASK_ATTACHMENTS, GET_ATTACHMENTS, = range(5)
SMTP_HOST = 'smtp-mail.outlook.com'
SMTP_PORT = 587
CON = None
FROM_ADDRESS = 'maniakbot@outlook.com'
FROM_ADDRESS_PASS = 'zasxcdfv20'
RECIPIENT_ADDRESS = None
SUBJECT = None
MESSAGE = None
ATTACHMENTS = None


# AIRBNB GLOBALS
AIRBNB_PARAMS = {}
AIRBNB_MODEL = None
NEIGHBOURHOOD, ROOM_TYPE, MINIMUM_NIGHTS = range(3)

# SEATTLE GLOBALS
SEATTLE_MODEL = None
SEATTLE_PARAMS = {}
SEATTLE_DF_DICT = {}
INCIDENT_TYPE, DISTRICT_SECTOR = range(2)

# TEXT ANALYTICS
BEST_PROFESSORS = range(1)
BEST_CONCEPT, BEST_GRADE, BALANCE = None, None, None

# MALARIA DETECTION
ACCEPT_CELL_DETECT = range(1)

# LOGGING
logging.root.removeHandler(absl.logging._absl_handler)
logfile_handler = logging.FileHandler('./bot.log')

absl.logging._warn_preinit_stderr = False
logging_format = '%(asctime)s - %(levelname)s - %(filename)s @%(lineno)d : %(funcName)s() - %(message)s'
logging_config = {
    'level': logging.INFO,
    'format': logging_format
}

logging.basicConfig(**logging_config)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(logging_format)
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def login_email():
    logger.info('Logging into mail server')
    global CON, SMTP_HOST, SMTP_PORT, FROM_ADDRESS, FROM_ADDRESS_PASS
    if not CON:
        start = time.time()
        CON = smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT)
        logger.info(f'Time taken to login is {time.time() - start} Secs')
        CON.starttls()
        success = CON.login(FROM_ADDRESS, FROM_ADDRESS_PASS)
        logger.info(f"Login {success}")
        return success
    return True


def msg_handler(update, context):
    text = update.message.text
    global BOT_CHAT_ID
    BOT_CHAT_ID = str(update.effective_chat.id)
    if text == 'Lock' or text == 'lock':
        ctypes.windll.user32.LockWorkStation()

    elif text.lower() in ['hello', 'hey', 'hi', 'hola', 'yo', 'heylo']:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"{text.capitalize()}! My name is {update.effective_message.bot.first_name}")

    elif text.lower() == 'facebook':
        wb.open('www.facebook.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Facebook")

    elif text.lower() == 'google':
        wb.open('www.google.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Google")

    elif text.lower() == 'youtube':
        wb.open('www.youtube.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Youtube")

    elif text.lower() == 'github':
        wb.open('www.github.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Github")

    elif text.lower() == 'quora':
        wb.open('www.quora.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Quora")

    elif text.lower() == 'linkedin':
        wb.open('www.linkedin.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened LinkedIn")

    elif text.lower() == 'elearning':
        wb.open('https://elearning.utdallas.edu/webapps/portal/execute/tabs/tabAction?tab_tab_group_id=_1_1')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened E-learning")

    elif text.lower() == 'balc':
        logger.info("Choose one to demonstrate:")
        logger.info("1. Airbnb Prediction")
        logger.info("2. Seattle Police PD Prediction")
        logger.info("3. Text Analytics UTD")
        logger.info("4. Image Classification")
        logger.info("5. Malaria Detection")

        update.message.reply_text(
            'Choose one for a demo:\n\n'
            '/airbnb_prediction\n\n'
            '/seattle_police_prediction\n\n'
            '/text_analytics\n\n'
            '/image_classification\n\n'
            '/malaria_detection\n\n')


def get_np_array(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return decoded


def classify(image_array):
    global MODEL, PREPROCESS_INPUT, GRAPH, SESS, INIT
    try:
        with GRAPH.as_default():
            set_session(SESS)
            # prepare image
            x = image_array
            # x = resize(x, (224, 224)) * 255
            # x = resize(x, (299, 299)) * 255
            x = resize(x, (331, 331)) * 255  # cast back to 0-255 range
            x = PREPROCESS_INPUT(x)
            x = np.expand_dims(x, 0)
            # processing image
            y = MODEL.predict(x)
            return decode_predictions(y)
    except Exception as e:
        logger.error(e)
        return None


def img_handler(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'll tell you what the image is in a moment.")
    photo_list = update.message['photo']
    file_id = photo_list[len(photo_list) - 1]['file_id']
    file_path = context.bot.getFile(file_id).file_path
    file = telegram.File(file_id, bot=context.bot, file_path=file_path)
    image_bytes = file.download_as_bytearray()
    image_np_array = get_np_array(image_bytes)
    predictions = classify(image_np_array)[0]
    if not predictions:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Couldn't figure that out. Try again later")
        return
    logger.info(predictions)
    best_pred = ' '.join([word.capitalize() for word in predictions[0][1].replace('_', ' ').split()])
    response = f'This is a {best_pred}. I am {predictions[0][2]*100:.2f}% confident'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def img_handler(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'll tell you what this is in a moment.")
    photo_list = update.message['photo']
    file_id = photo_list[len(photo_list) - 1]['file_id']
    file_path = context.bot.getFile(file_id).file_path
    file = telegram.File(file_id, bot=context.bot, file_path=file_path)
    image_bytes = file.download_as_bytearray()
    # image = file.download('sample.png')
    image_np_array = get_np_array(image_bytes)
    predictions = classify(image_np_array)[0]
    if not predictions:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Couldn't figure that out. Try again later")
        return
    logger.info(predictions)
    best_pred = ' '.join([word.capitalize() for word in predictions[0][1].replace('_', ' ').split()])
    response = f'This is a {best_pred}. I am {predictions[0][2]*100:.2f}% confident'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Conversation ended',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def send_to(update, context):
    update.message.reply_text(
        'Enter the E-mail address of the recipient')

    return RECIPIENT


def to_address(update, context):
    global RECIPIENT_ADDRESS
    RECIPIENT_ADDRESS = update.message.text
    update.message.reply_text(
        'Enter the E-mail subject')

    return ASK_SUBJECT


def email_subject(update, context):
    global SUBJECT
    SUBJECT = update.message.text
    update.message.reply_text(
        'Enter the E-mail message')
    return ASK_MESSAGE


def email_message(update, context):
    global MESSAGE
    MESSAGE = update.message.text
    # if send_email():
    #     update.message.reply_text(
    #         'Email sent successfully!')
    # else:
    #     update.message.reply_text(
    #         'Failed to send message')
    reply_keyboard = [['Yes', 'No']]
    update.message.reply_text(
            'Do you want to attach some files?', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return ASK_ATTACHMENTS


def ask_attachments(update, context):
    reply = update.message.text
    if reply == 'Yes':
        update.message.reply_text(
            'Send attachments')
        return GET_ATTACHMENTS
    else:
        if send_email():
            update.message.reply_text(
                'Email sent successfully')
        else:
            update.message.reply_text('Sending email failed')
        return ConversationHandler.END  # Ends conversation


def email_attachments(update, context):
    global ATTACHMENTS
    types = ['photo', 'audio', 'document', 'video']
    file_types_found = []
    file_bytes_list = {}
    for t in types:
        if update.message[t]:
            file_types_found.append(t)
    for file_type in file_types_found:
        if file_type == 'photo':
            file_id = update.message[file_type][len(update.message[file_type])-1]['file_id']
            file_name = update.message[file_type][len(update.message[file_type])-1]['file_id'] + '.jpg'
        else:
            file_id = update.message[file_type]['file_id']
            file_name = update.message[file_type]['file_name']
        file_path = context.bot.getFile(file_id).file_path
        file = telegram.File(file_id, bot=context.bot, file_path=file_path)
        file_bytes = bytes(file.download_as_bytearray())
        file_bytes_list[file_name] = file_bytes
    if send_email(files=file_bytes_list):
        update.message.reply_text(
            'Email sent successfully')
    else:
        update.message.reply_text(
            'Sending email failed')
    return ConversationHandler.END


def send_email(files: 'dict' = None) -> 'bool':
    login_email()
    global FROM_ADDRESS, RECIPIENT_ADDRESS, SUBJECT, MESSAGE, CON
    msg = MIMEMultipart()
    msg['From'] = FROM_ADDRESS
    msg['To'] = RECIPIENT_ADDRESS
    msg['Subject'] = SUBJECT
    msg.attach(MIMEText(MESSAGE, 'plain'))
    if files:
        for filename, file in files.items():
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            msg.attach(part)
    try:
        CON.send_message(msg)
        CON.quit()
        CON = None
        return True
    except Exception as e:
        logger.error(e)
        return False


# COMMAND HANDLERS FROM HERE


def neighbourhood(update, context):
    global AIRBNB_PARAMS
    AIRBNB_PARAMS['neighbourhood_group'] = update.message.text
    reply_keyboard = [
        ['Private room', 'Shared room'],
        ['Entire home/apt']
    ]
    update.message.reply_text(
        'What kind of room are you looking for?', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))

    return ROOM_TYPE


def room_type(update, context):
    global AIRBNB_PARAMS
    AIRBNB_PARAMS['room_type'] = update.message.text
    update.message.reply_text(
        'How many nights are you planning on staying?')

    return MINIMUM_NIGHTS


def minimum_nights(update, context):
    global AIRBNB_PARAMS
    AIRBNB_PARAMS['minimum_nights'] = [update.message.text]
    result = predict_airbnb()
    update.message.reply_text(result)
    return ConversationHandler.END


def predict_airbnb():
    global AIRBNB_MODEL, AIRBNB_PARAMS
    neighbourhood_mapping = {}
    neighbourhood_mapping['Brooklyn'], neighbourhood_mapping['Manhattan'], neighbourhood_mapping['Queens'],\
    neighbourhood_mapping['Staten Island'], neighbourhood_mapping['Bronx'] = range(5)
    room_type_mapping = {}
    room_type_mapping['Private room'], room_type_mapping['Entire home/apt'], room_type_mapping['Shared room'] = range(3)
    AIRBNB_PARAMS['neighbourhood_group'] = [neighbourhood_mapping[AIRBNB_PARAMS['neighbourhood_group']]]
    AIRBNB_PARAMS['room_type'] = [room_type_mapping[AIRBNB_PARAMS['room_type']]]
    airbnb_df = pd.DataFrame.from_dict(AIRBNB_PARAMS)
    predictions = AIRBNB_MODEL.predict(airbnb_df)
    logging.info(predictions)
    return f"Your budget should be close to ${predictions[0]:.2f}"


def airbnb_prediction(update, context):
    reply_keyboard = [
        ['Manhattan', 'Queens'],
        ['Bronx', 'Brooklyn'],
        ['Staten Island']
    ]
    update.message.reply_text(
        'Which neighbourhood are you looking to live in?', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return NEIGHBOURHOOD


def incident_type(update, context):
    global SEATTLE_PARAMS
    SEATTLE_PARAMS['incident_type'] = update.message.text
    reply_keyboard = [
        ['D', 'E', 'F', 'J', 'K'],
        ['L', 'M', 'N', 'Q', 'R'],
        ['S', 'U', 'W', 'Other']
    ]
    update.message.reply_text(
        'Which sector was that in?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return DISTRICT_SECTOR


def district_sector(update, context):
    global SEATTLE_PARAMS
    SEATTLE_PARAMS['district_sector'] = update.message.text
    result = predict_seattle()
    update.message.reply_text(
        result)
    return ConversationHandler.END



def predict_seattle():
    global SEATTLE_PARAMS, SEATTLE_DF_DICT, SEATTLE_MODEL
    incident = None
    sector = None
    for key, value in SEATTLE_DF_DICT.items():
        col_to_check = key.split('_')[1]
        if SEATTLE_PARAMS['incident_type'].lower() in col_to_check.lower():
            SEATTLE_DF_DICT[key] = 1
            incident = col_to_check
        if SEATTLE_PARAMS['district_sector'] == col_to_check:
            SEATTLE_DF_DICT[key] = 1
            sector = col_to_check
    seattle_df = pd.DataFrame.from_dict(SEATTLE_DF_DICT)
    predictions = SEATTLE_MODEL.predict(seattle_df)
    logging.info(f'Prediction is {predictions}')
    pred_val = str(predictions[0][0])
    hours, mins = pred_val.split('.')
    return f"{incident} incident(s) in Sector {sector} is going to take {hours} hours {(float('0.'+mins)*60):.1f} minutes to be cleared"


def seattle_police_prediction(update, context):
    reply_keyboard = [
        ['Theft', 'Residential Burglaries'],
        ['Traffic related', 'Suspicious Circumstances'],
        ['Other']
    ]
    update.message.reply_text(
        'Which type of incident is it?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return INCIDENT_TYPE


def text_analytics(update, context):
    # images_dir = 'text_analytics_images'
    # for image in os.listdir(images_dir):
    #     context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(images_dir+'/'+image, 'rb'))
    reply_keyboard = [
        ['Good concepts', 'Good grades'],
        ['Balance of both']
    ]
    update.message.reply_text(
        'Which kind of professors are you looking for?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return BEST_PROFESSORS


def best_professors(update, context):
    if 'concept' in update.message.text.lower():
        result = '\n\u2022'.join(BEST_CONCEPT.tolist())
    elif 'grade' in update.message.text.lower():
        result = '\n\u2022'.join(BEST_GRADE.tolist())
    elif 'balance' in update.message.text.lower():
        result = '\n\u2022'.join(BALANCE.tolist())

    update.message.reply_text(
        f'For {update.message.text} you could choose the following professors:\n\u2022{result}')
    return ConversationHandler.END


def malaria_detect_entry(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Send me an image of blood smear cell")

    return ACCEPT_CELL_DETECT


def detect_malaria(update, context):
    global FASTAI_MODEL
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'll tell you if this is infected in a moment.")
    photo_list = update.message['photo']
    file_id = photo_list[len(photo_list) - 1]['file_id']
    file_path = context.bot.getFile(file_id).file_path
    file = telegram.File(file_id, bot=context.bot, file_path=file_path)
    image = file.download('sample.png')
    img = open_image('sample.png')
    pred_class, b, c = FASTAI_MODEL.predict(img)
    response = f'This is {pred_class}.'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    return ConversationHandler.END


def image_classification(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Send me an image")


def load_handlers():
    handlers = []
    email_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.regex('^(email|Email|E-mail|e-mail)$'), send_to)],
        # Triggers sending an Email conversation

        states={
            RECIPIENT: [
                MessageHandler(Filters.regex("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$)"), to_address)],

            ASK_SUBJECT: [MessageHandler(Filters.text, email_subject)],

            ASK_MESSAGE: [MessageHandler(Filters.text, email_message)],

            ASK_ATTACHMENTS: [MessageHandler(Filters.text, ask_attachments)],

            GET_ATTACHMENTS: [
                MessageHandler(Filters.document | Filters.photo | Filters.video | Filters.audio | Filters.contact,
                               email_attachments)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )
    handlers.append(email_conv_handler)

    airbnb_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('airbnb_prediction', airbnb_prediction)],

        states={
            NEIGHBOURHOOD: [
                MessageHandler(Filters.text, neighbourhood)],

            ROOM_TYPE: [MessageHandler(Filters.text, room_type)],

            MINIMUM_NIGHTS: [MessageHandler(Filters.text, minimum_nights)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )
    handlers.append(airbnb_conv_handler)

    seattle_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('seattle_police_prediction', seattle_police_prediction)],
        states={
            INCIDENT_TYPE: [
                MessageHandler(Filters.text, incident_type)],

            DISTRICT_SECTOR: [MessageHandler(Filters.text, district_sector)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )
    handlers.append(seattle_conv_handler)

    text_anal_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('text_analytics', text_analytics)],
        states={
            BEST_PROFESSORS: [
                MessageHandler(Filters.text, best_professors)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )
    handlers.append(text_anal_conv_handler)

    malaria_detect_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('malaria_detection', malaria_detect_entry)],
        states={
            ACCEPT_CELL_DETECT: [
                MessageHandler(Filters.photo, detect_malaria)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )
    handlers.append(malaria_detect_conv_handler)

    handlers.append(MessageHandler(Filters.text, msg_handler))  # General text handler for greetings and BALC options
    handlers.append(MessageHandler(Filters.photo, img_handler))  # Handler to filter photos for image classification
    handlers.append(CommandHandler('image_classification', image_classification))
    handlers.append(CommandHandler('weather', weather))
    handlers.append(MessageHandler(Filters.location, location))
    return handlers


def location(update, context):
    latitude = update.message.location.latitude
    longitude = update.message.location.longitude
    base_url = 'http://api.openweathermap.org/data/2.5/weather?'
    url = base_url + "appid=" + WEATHER_KEY + "&lat=" + str(latitude) + "&lon="+str(longitude)
    response = requests.get(url).json()
    if response['cod'] == 200:
        result = f"It's {response['weather'][0]['main']} with temperature at {((response['main']['temp']*9/5)-459.67):.2f} °F in {response['name']}"
    update.message.reply_text(
        result)


def weather(update, context):
    location_keyboard = telegram.KeyboardButton(text="send_location", request_location=True)
    update.message.reply_text(
        'What',
        reply_markup=ReplyKeyboardMarkup([[location_keyboard]], one_time_keyboard=True, resize_keyboard=True))


def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    # Add all handlers to the dispatcher
    for handler in load_handlers():  # Adding all handlers
        dp.add_handler(handler)
    dp.add_error_handler(error)
    logger.info('Polling for updates. Start chatting')
    updater.start_polling()
    updater.idle()


def load_model():
    global MODEL, PREPROCESS_INPUT, GRAPH, SESS, INIT, MODEL_TO_FETCH
    start = time.time()
    logger.info(f'Loading model {MODEL_TO_FETCH}')
    SESS = tf.Session()
    GRAPH = tf.get_default_graph()
    set_session(SESS)
    # INIT = tf.global_variables_initializer()
    init_model, PREPROCESS_INPUT = Classifiers.get(MODEL_TO_FETCH)
    # MODEL = init_model(input_shape=(224, 224, 3), weights='imagenet', classes=1000)
    # MODEL = init_model(input_shape=(299, 299, 3), weights='imagenet', classes=1000)
    MODEL = init_model(input_shape=(331, 331, 3), weights='imagenet', classes=1000)
    logger.info(f'Loaded model {MODEL_TO_FETCH}')
    logger.info(f'Time taken to load {MODEL_TO_FETCH} model is {time.time() - start} Secs')

def load_seattle_model():
    global SEATTLE_MODEL, SEATTLE_DF_DICT
    start = time.time()
    SEATTLE_MODEL = joblib.load('SeattleModel (1).pkl')
    logger.info(f'Time taken to load seattle model is {time.time() - start} Secs')
    columns = ['District/Sector_D', 'District/Sector_E', 'District/Sector_F', 'District/Sector_J'
        , 'District/Sector_K', 'District/Sector_L', 'District/Sector_M',
               'District/Sector_N', 'District/Sector_Other', 'District/Sector_Q'
        , 'District/Sector_R', 'District/Sector_S', 'District/Sector_U'
        , 'District/Sector_W', 'Initial Type Group_Other',
               'Initial Type Group_RESIDENTIAL BURGLARIES'
        , 'Initial Type Group_SUSPICIOUS CIRCUMSTANCES',
               'Initial Type Group_THEFT', 'Initial Type Group_TRAFFIC RELATED CALLS']
    SEATTLE_DF_DICT = {}
    for col in columns:
        SEATTLE_DF_DICT[col] = [0]


def load_airbnb_model():
    global AIRBNB_MODEL
    start = time.time()
    AIRBNB_MODEL = joblib.load('model (2).pkl')
    logger.info(f'Time taken to load Airbnb model is {time.time() - start} Secs')


def load_text_anal():
    global BEST_CONCEPT, BEST_GRADE, BALANCE
    df = pd.read_csv('responses.csv', encoding="ISO-8859-1")
    df = df.drop(['Timestamp'], axis=1)
    df['Grading_Score'] = df['Rate the Professor on basis of grading']
    df['Concept_Score'] = df['Rate the Professor on basis of teaching']
    df['Grading_Score'] = df['Grading_Score'].map({'Easy': 5, 'Average': 3, 'Very tough': 1})
    df['Concept_Score'] = df['Concept_Score'].map({'Excellent': 5, 'Very Good': 3, 'Good': 1})
    df = df.drop(['Rate the Professor on basis of grading', 'Rate the Professor on basis of teaching'], axis=1)
    table = pd.pivot_table(df, values=['Grading_Score', 'Concept_Score'], index=['Professor Name'], aggfunc=np.average)
    table['Balanced_Score'] = (table['Grading_Score'] + table['Concept_Score']) / 2
    table.reset_index(level=0, inplace=True)
    concept_data = table.sort_values(by=['Concept_Score'], ascending=False).head(5)
    BEST_CONCEPT = concept_data['Professor Name']
    grade_data = table.sort_values(by=['Grading_Score'], ascending=False).head(5)
    BEST_GRADE = grade_data['Professor Name']
    balance_data = table.sort_values(by=['Balanced_Score'], ascending=False).head(5)
    BALANCE = balance_data['Professor Name']


def load_fastai_model():
    global FASTAI_MODEL
    img_dir = './cell_images/'
    path = Path(img_dir)
    data = ImageDataBunch.from_folder(path, train=".",
                                      valid_pct=0.2,
                                      ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                      size=224, bs=64,
                                      num_workers=0).normalize(imagenet_stats)
    print('DATA IS ', data)
    FASTAI_MODEL = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/").load('stage-2')


if __name__ == '__main__':
    # logger.info('Clearing keras session')
    # keras.backend.clear_session()
    load_model()
    load_seattle_model()
    load_airbnb_model()
    load_text_anal()
    load_fastai_model()
    main()