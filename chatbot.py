import random
import json
import pickle
import numpy as np
import nltk
import math
from nltk.stem import WordNetLemmatizer
from keras.models import load_model 

class chatbotAI:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('intents.json', encoding="utf-8").read())

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.h5')

        self.stateSpace = ["default", "typing coeff"]

        self.state = "default"

        self.inputNum = None
        self.arr = None
        self.current_coeffsNum = 0

        print("GO! Bot is running!")


    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words (self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class (self, sentence):
        bow = self.bag_of_words (sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes [r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        print(intents_list)
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
        return result


    
    def chatAPI(self, msg):
        intents_list = self.predict_class (msg)
        tag = intents_list[0]['intent']
        inputNumCases = ["math1", "math2", "math3"]

        match self.stateSpace.index(self.state)+1:
            case 1:
                
                # handle state transition
                if(tag in inputNumCases):
                    
                    self.inputNum = inputNumCases.index(tag) + 2
                    self.arr = np.zeros(self.inputNum)

                    # assign new state
                    self.state = self.stateSpace[ self.stateSpace.index(self.state)+1 ]
                    return "điền hệ số đi bro"

                else:
                    ints = self.predict_class (msg)
                    res = self.get_response (ints, self.intents)
                    return res

            case 2:
                if self.current_coeffsNum < self.inputNum:

                    self.arr[self.current_coeffsNum] = int(msg)

                    self.current_coeffsNum = self.current_coeffsNum + 1


                if self.current_coeffsNum == self.inputNum:
               
                    # assign new state
                    self.state = self.stateSpace[ self.stateSpace.index(self.state)-1 ]
                    
                    return self.solveEq(self.arr)
                else:
                    return "Nhập tiếp đi bro"

            case _:
                res = self.get_response (ints, self.intents)
                return res
