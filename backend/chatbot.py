import os
import json
import pickle
import random
import nltk
import numpy
from nltk.stem import LancasterStemmer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


from kivymd.app import MDApp
from kivymd.uix.screenmanager import ScreenManager
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.button import *
from kivymd.uix.label import *
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import StringProperty, NumericProperty
from kivy.core.text import LabelBase
from kivy.clock import Clock


Window.size = (350, 550)

nltk.download('punkt')

stemmer = LancasterStemmer()

with open("backend\intents.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
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
    yaml_file = open('chatbotmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_yaml(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # Make our neural network
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # optimize the model
    myChatModel.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

    # train the model
    myChatModel.fit(training, output, epochs=1000, batch_size=8)

    # serialize model to yaml and save it to disk
    model_yaml = myChatModel.to_json()
    with open("chatbotmodel.yaml", "w") as y_file:
        y_file.write(model_yaml)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chatWithBot(inputText):
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        return "I didn't get that, try again"

    result = myChatModel.predict(numpyCurrentText[0:1])
    result_index = numpy.argmax(result)
    tag = labels[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    else:
        return "I didn't get that, try again"


def chat(self):
    print("Start talking with the chatbot (try quit to stop)")
    inp = value
    #x = 1
    print(chatWithBot(inp))
    #print("printing here also")
    #show = ChatBot()
    global answer
    answer = chatWithBot(inp)


class Command(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name: "fonts\Poppins-Medium.ttf"
    font_size = 17


class Response(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name: "fonts\Poppins-Medium.ttf"
    font_size = 17


class ChatBot(MDApp):

    def chat(self):
        print("Start talking with the chatbot (try quit to stop)")
        inp = value
        #x = 1
        print(chatWithBot(inp))
        #print("printing here also")
        show = ChatBot()
        global answer
        answer = chatWithBot(inp)

    def change_screen(self, name):
        screen_manager.set_current("name")

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("kv\\Chats.kv"))
        screen_manager.add_widget(Builder.load_file("kv\\Main.kv"))
        return screen_manager

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
            chat(value)
            screen_manager.get_screen('chats').chat_list.add_widget(
                Command(text=value, size_hint_x=size, halign=halign))
            Clock.schedule_once(self.response, 1)
            screen_manager.get_screen('chats').text_input.text = ""


if __name__ == '__main__':
    app = ChatBot()
    app.run()
