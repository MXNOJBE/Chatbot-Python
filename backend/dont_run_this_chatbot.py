import os
import json
import pickle
import random
import nltk
import numpy
from nltk.stem import LancasterStemmer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json
from kivymd.app import MDApp
from kivymd.uix.screenmanager import ScreenManager
from kivymd.uix.label import MDLabel
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty
from kivy.clock import Clock

Window.size = (350, 550)

stemmer = LancasterStemmer()

with open("backend/intents.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

try:
    with open('chatbotmodel.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        myChatModel = model_from_json(loaded_model_json)
        myChatModel.load_weights("chatbotmodel.h5")
        print("Loaded model from disk")

except FileNotFoundError:
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    myChatModel.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

    myChatModel.fit(training, output, epochs=1000, batch_size=8)

    model_json = myChatModel.to_json()
    with open("chatbotmodel.json", "w") as json_file:
        json_file.write(model_json)

    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model to disk")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat_with_bot(input_text):
    current_text = bag_of_words(input_text, words)
    current_text_array = [current_text]
    numpy_current_text = numpy.array(current_text_array)

    if numpy.all((numpy_current_text == 0)):
        return "I didn't get that, try again"

    result = myChatModel.predict(numpy_current_text[0:1])
    result_index = numpy.argmax(result)
    tag = labels[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    else:
        return "I didn't get that, try again"


class Command(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name = "fonts/Poppins-Medium.ttf"
    font_size = 17


class Response(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name = "fonts/Poppins-Medium.ttf"
    font_size = 17


class ChatBotApp(MDApp):

    def chat(self, value):
        print("Start talking with the chatbot (try quit to stop)")
        inp = value
        print(chat_with_bot(inp))
        global answer
        answer = chat_with_bot(inp)

    def change_screen(self, name):
        screen_manager.set_current(name)

    def response(self, *args):
        if len(answer) < 6:
            size = .22
            halign = "center"
        elif len(answer) < 11:
            size = .32
            halign = 'center'
        elif len(answer) < 16:
            size = .45
            halign = 'center'
        elif len(answer) < 21:
            size = .58
            halign = 'center'
        elif len(answer) < 26:
            size = .71
            halign = 'center'
        else:
            size = .77
            halign = "left"
        screen_manager.get_screen('chats').chat_list.add_widget(
            Response(text=answer, size_hint_x=.75, halign=halign))

    def send(self):
        global size, halign, value
        if screen_manager.get_screen('chats').text_input != "":
            value = screen_manager.get_screen('chats').text_input.text
            if len(value) < 6:
                size = .22
                halign = "center"
            elif len(value) < 11:
                size = .32
                halign = 'center'
            elif len(value) < 16:
                size = .45
                halign = 'center'
            elif len(value) < 21:
                size = .58
                halign = 'center'
            elif len(value) < 26:
                size = .71
                halign = 'center'
            else:
                size = .77
                halign = "left"
            self.chat(value)
            screen_manager.get_screen('chats').chat_list.add_widget(
                Command(text=value, size_hint_x=size, halign=halign))
            Clock.schedule_once(self.response, 1)
            screen_manager.get_screen('chats').text_input.text = ""


if __name__ == '__main__':
    app = ChatBotApp()
    app.run()
