import os
import json
import pickle
import random
import nltk
import numpy
import requests
from nltk.stem import LancasterStemmer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json

base_url = "https://ergast.com/api/f1"

stemmer = LancasterStemmer()

# Fetching seasons data from the Ergast API
seasons_url = f"{base_url}/seasons.json"
response = requests.get(seasons_url)
seasons_data = response.json()

# Updating data with seasons information
data = seasons_data

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
        if tag == "ergast_data":
            return handle_ergast_api_request(input_text)
        else:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)

    else:
        return "I didn't get that, try again"


def handle_ergast_api_request(input_text):
    # Extract necessary information from the input_text
    # and make a request to the Ergast API
    driver_url = "https://ergast.com/api/f1/drivers.json?limit=858&offset=0"
    driver_response = requests.get(driver_url).json()
    driver_data = driver_response['MRData']['DriverTable']['Drivers']

        # Extract relevant information from driver_data
    for driver in driver_data:
        if(driver['givenName'] == input_text):
            print(f'driver name you have gievn is: {input_text}')
            response_text = f"{driver['givenName']} {driver['familyName']} is a {driver['nationality']} driver."
            print(response_text)

        else:
            response_text = "Sorry, I couldn't find information about that driver."

    
    return response_text


if __name__ == '__main__':
    print("Start talking with the chatbot (type 'quit' to stop)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        else:
            response = handle_ergast_api_request(user_input)
            print("Bot:", response)
