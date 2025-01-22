

import cv2
import numpy as np
from scipy.spatial.distance import cosine
import os

from telegram import *
import asyncio
import telegram.error
from telegram.ext import * 

# Load face detection model
config = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(config, model)
cv_image = None

# Load ArcFace ONNX model
arcface_model = cv2.dnn.readNetFromONNX("arcface.onnx")

token = "YOUR TOKEN"
bot = Bot(token=token)
bot_username = "YOUR BOT'S USERNAME"


#                   ============= TELEGRAM COMMANDS ============== 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a message with buttons."""
    buttons = [
        [InlineKeyboardButton("List registereed students", callback_data="list_registered_students")],
        [InlineKeyboardButton("Check attendance (list)", callback_data="list_students")],
        [InlineKeyboardButton("Photo of the class", callback_data="image_retrieve")]
        ]
    print("/start used by", update.effective_user.first_name.strip(), update.effective_user.id)
    await update.message.reply_text("Choose an option:", reply_markup=InlineKeyboardMarkup(buttons))

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("/about used by", update.effective_user.first_name.strip(), update.effective_user.id)    
    await context.bot.sendMessage(chat_id=update.effective_user.id, text="This bot is developed on PTB (Python-Telegram-Bot), using additional tools like OpenCV's DNN module for facial recognition and cosine formula for similarity calculations.")

async def list_registered_students(update: Update, context: CallbackContext):
    table = student_names
    txt = ''
    for i in range(len(table)):
        txt += f'{table[i]}\n'
    print(txt)
    print("/list_registered_students used by", update.effective_user.first_name.strip(), update.effective_user.id)
    await bot.sendMessage(chat_id=update.effective_user.id, text=txt)
    
async def list_students(update: Update, context: ContextTypes.DEFAULT_TYPE):

    unknown_faces_count = 0
    student_status =  {name: '-' for name in student_names}

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            # Get embedding for detected face
            embedding = get_face_embedding(face)

            match, max_similarity = analyze_similarity(embedding, known_faces, threshold=0.3)
            # Annotate the frame
            if max_similarity>0.3:
                student_status[match] = "+"
            else:
                unknown_faces_count += 1
    table = [(key, value) for key, value in student_status.items()]
    table.append(('Not recognized:', unknown_faces_count))
    txt = ''
    for i in range(len(table)):
        txt += f'{table[i][0]}  {table[i][1]}\n'
    print(txt)
    print("/list_students used by", update.effective_user.first_name.strip(), update.effective_user.id)
    await bot.sendMessage(chat_id=update.effective_user.id, text=txt)

async def image_retrieve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    # scanning_message = await context.bot.sendMessage(chat_id=update.effective_user.id, text="Scanning...")
    """Captures a photo from the webcam, detects faces, and sends the result."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await update.message.reply_text("Error: Unable to access the camera. Please check your device.")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        await update.message.reply_text("Error: Unable to capture an image. Please try again.")
        cap.release()
        return

    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            # Get embedding for detected face
            embedding = get_face_embedding(face)

            match, max_similarity = analyze_similarity(embedding, known_faces, threshold=0.3)
            # Annotate the frame
            if max_similarity>0.3:
                text = f"{match}: {max_similarity*100:.2f}%"
                color = (0, 255, 0)
            else:
                text = f"Unknown: {max_similarity*100:.0f}% similarity with {match}"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check if the frame has content before encoding
    if frame is None or frame.size == 0:
        await update.message.reply_text("Error: Captured frame is empty. Please try again.")
        cap.release()
        return

    # Encode the image and send it back to the user
    _, buffer = cv2.imencode(".jpg", frame)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer.tobytes())
    print("/image_retrieve used by", update.effective_user.first_name.strip(), update.effective_user.id)
    cap.release()
    cv2.destroyAllWindows()
    start(update, context)




#               ============== MESSAGE/BUTTON/ERROR HANDLERS ============== 

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles button clicks."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button click

    # Call specific functions based on callback data
    # match query.data:
    if query.data == "list_registered_students":
            await list_registered_students(update, context)
    elif query.data == "list_students":
            await list_students(update, context)
    elif query.data == "image_retrieve":
            await image_retrieve(update, context)

async def handle_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type:str = update.message.chat.type    #--dialogue or group
    text = update.message.text
    print(f'\n\n__________________\nUser {update.message.from_user.first_name}({update.message.chat.id}) in {message_type} \n=====\nMessage: {text}\n=====')
    if message_type=="group":
        if bot_username in text:
            new_text = text.replace(bot_username, '').strip()
            response = handle_response(new_text)
    else:
        new_text = text
        response = handle_response(new_text)
    print(f'Bot: {response}')
    await bot.sendMessage(chat_id=update.effective_user.id, text=response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'\n\n________________\nUpdate: {update}\n=====\nError: {context.error}\n=====\nContext: {context}')





#                  ============ AUXILIARY FUNCTIONS ===================
#         (to simplify the repeated usage of the same code in async functions)

def handle_response(text):
    if ('hello' in text) or ('Hello' in text):
        return 'Hey there!'
    else:
        return 'This bot is not designed for chatting ;) \nplease, use /start'

def get_face_embedding(face_image):
    # Preprocess face: resize to 112x112, normalize, etc.
    face_image = cv2.resize(face_image, (112, 112))
    face_image = face_image.astype("float32") / 255.0
    blob = cv2.dnn.blobFromImage(face_image)
    arcface_model.setInput(blob)
    embedding = arcface_model.forward()
    return embedding.flatten()

# Load known faces
def load_known_faces(folder="known_faces"):
    known_faces = {}
    for name in os.listdir(folder):
        person_folder = os.path.join(folder, name)
        if os.path.isdir(person_folder) and not name.startswith('.'):
            embeddings = []
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                embedding = get_face_embedding(image)
                embeddings.append(embedding)
            known_faces[name] = embeddings
    return known_faces

def get_student_names(folder="known_faces"):
    folder_path = os.path.join(os.getcwd(), folder)  # Get the full path of the folder
    if not os.path.exists(folder_path):
        print(f"Folder '{folder}' does not exist!")
        return []

    student_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return student_names

student_names = get_student_names()
known_faces = load_known_faces()

def analyze_similarity(embedding, known_faces, threshold=0.3):
    match = None
    max_similarity = 0
    for name, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = 1 - cosine(embedding, known_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                match = name
    return match, max_similarity




#          ================ THE MAIN PART ==================
#                    (a.k.a. builder and polling)
def main():

    # Commands
    application = Application.builder().token(token=token).build()
    application.add_handler(CommandHandler(command='about', callback=about))
    application.add_handler(CommandHandler(command='list_registered_students', callback=list_registered_students))
    application.add_handler(CommandHandler(command='start', callback=start))
    application.add_handler(CommandHandler(command='list_students', callback=list_students))
    application.add_handler(CommandHandler(command='image_retrieve', callback=image_retrieve))
        
    # Messages
    application.add_handler(MessageHandler(filters.TEXT, callback=handle_messages))

    #Button handler
    application.add_handler(CallbackQueryHandler(handle_button_click))

    # Errors
    application.add_error_handler(error)
    
    
    print('\n\ntelegram bot launched successfully\n\n')
    application.run_polling(poll_interval=0.5)

if __name__ == '__main__':
    main()

# Detect and recognize faces in live feed
