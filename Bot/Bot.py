import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

import tensorflow as tf

import tflearn
import random
import json


class Bot:

    def __init__(self, updating_model = False, epochs = 2000):
        self.stemmer = LancasterStemmer()
        labels, words, training, output = data_preperation(self.stemmer, renew=updating_model)
        self.labels = labels
        self.words = words
        self.training = training
        self.output = output
        self.model = get_model(self.training, self.output, updating_model, epochs=epochs)

    def get_sentences_words_output(self, sentence):
        bag = [0 for x in range(len(self.words))]

        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(w.lower()) for w in sentence_words]

        for s_word in sentence_words:
            for i, word in enumerate(self.words):
                if word == s_word:
                    bag[i] = 1

        return np.array(bag)

    def create_respone(self, sentence):
        preds = self.model.predict([self.get_sentences_words_output(sentence)])
        if preds[0][np.argmax(preds)] > 0.6:
            print(preds[0][np.argmax(preds)])
            label = self.labels[np.argmax(preds)]

            for i in get_intents():
                if i["tag"] == label:
                    responses = i["responses"]
                    return 'Answer : {}'.format(responses[random.randint(0, len(responses) - 1)])
        else:
            print(preds[0][np.argmax(preds)])
            return "I can't actually uderstand what did you said.\nCan you repeat it please..."


def data_preperation(stemmer, renew = False):
    try:
        if renew is True:
            x = 1/0
            print("=============Renewing=================")

        with open("../Bot/datas.json", "r") as f:
            return json.load(f)["datas"]
    except:
        with open('../Bot/intents.json') as file:
            data = json.load(file)

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

        words = [stemmer.stem(w.lower()) for w in words if w != '?']
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        output_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = list(output_empty)
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)
        diction = {"datas":[labels, words, training, output]}

        with open("../Bot/datas", "w") as f:
            json.dump(diction, f)

        return diction["datas"]


def set_model(training, output, model, fitting = False, epochs = 2000):
    try:
        if fitting is True:
            x = 1/0

        model.load("../Bot/model.tflearn")

    except:
        model.fit(training, output, n_epoch=epochs, batch_size=1, show_metric=True)
        model.save("../Bot/model.tflearn")



def get_model(training, output, renew_fitting = False, epochs = 2000):
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 100)
    net = tflearn.fully_connected(net, 100)
    net = tflearn.dropout(net, 0.75)
    net = tflearn.fully_connected(net, 20)
    net = tflearn.dropout(net,0.75)
    net = tflearn.fully_connected(net, 20)
    net = tflearn.dropout(net, 0.75)
    net = tflearn.fully_connected(net, 20)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    set_model(training, output, model, fitting=renew_fitting, epochs= epochs)
    return model


def get_intents():
    with open("../Bot/intents.json", "r") as f:
        return json.load(f)["intents"]


def conversation(bot):
    s = input("enter a sentence...")
    while s.lower() != "quit":
        print(bot.create_respone(s))
        s = input("\nenter a sentence...")


