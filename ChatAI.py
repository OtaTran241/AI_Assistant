import requests
import pyttsx3
import numpy as np
import random
import nltk
# from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from nltk import sent_tokenize
import re
from os.path import join, dirname, abspath
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# nltk_data_dir = 'nltk_data'
# nltk.data.path.append(nltk_data_dir)

# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('omw-1.4', download_dir=nltk_data_dir)
# nltk.download('wordnet', download_dir=nltk_data_dir)

lemmatizer = WordNetLemmatizer()

engine = pyttsx3.init()

current_dir = dirname(abspath(__file__))

with open(join(current_dir, 'Models/ChatbotData.pkl'), 'rb') as pickle_file:
    intents = pickle.load(pickle_file)

words = pickle.load(open(join(current_dir, 'Models/words.pkl'),'rb'))

classes = pickle.load(open(join(current_dir, 'Models/classes.pkl'),'rb'))

model = load_model(join(current_dir, 'Models/chatbot_model.h5'))

def google_search(query, api_key, cse_id, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'num': num_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        return results['items']
    else:
        print(f"Error: {response.status_code}")
        return []
    
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)

    for s in sentence_words:  
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)

    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

def getResponse(ints, intents_json):
    result = "none"
    if len(ints)==0:
        tag='No answer!!'
    else:    
        tag=ints[0]['intent']

    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result, tag

def chatbot_response(text): 
    ints = predict_class(text, model)

    res, tag = getResponse(ints, intents)

    return res, tag
    

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    return content

def get_google_search(test):
    search_results = google_search(test, api_key, s_id)

    return search_results

def speech(text):
    engine.say(text)

    engine.runAndWait()

def getContent(url):
    try:
        html = requests.get(url, timeout = 4)
        tree = BeautifulSoup(html.text,'lxml')
        for invisible_elem in tree.find_all(['script', 'style']):
            invisible_elem.extract()

        paragraphs = [p.get_text() for p in tree.find_all("p")]

        for para in tree.find_all('p'):
            para.extract()

        for href in tree.find_all(['a','strong']):
            href.unwrap()

        tree = BeautifulSoup(str(tree.html),'lxml')

        text = tree.get_text(separator='\n\n')
        text = re.sub('\n +\n','\n\n',text)

        paragraphs += text.split('\n\n')
        paragraphs = [re.sub(' +',' ',p.strip()) for p in paragraphs]
        paragraphs = [p for p in paragraphs if len(p.split()) > 10]

        for i in range(0,len(paragraphs)):
            sents = []
            text_chunks = list(chunks(paragraphs[i],5000))
            for chunk in text_chunks:
                sents += sent_tokenize(chunk)

            sents = [s for s in sents if len(s) > 2]
            sents = ' . '.join(sents)
            paragraphs[i] = sents

        return '\n\n'.join(paragraphs)
    except:
        return ''

def try_get_content(search_results):
    for result in search_results:
        content = getContent(result['link'])
        if content:
            return result['link'] , content
    return None, None

api_key = read_file(join(current_dir, 'API_Key.txt'))
s_id = read_file(join(current_dir, 'S_id.txt'))
# search_results = google_search(query, api_key, s_id)

# for i, item in enumerate(search_results):
#     print(f'\033[1;33m Bạn: {query}')
#     print(f"\033[1;32m Bot: {item['title']}")
#     print(f"\033[1;32m URL: {item['link']}")
#     print(item)
#     speech(res)
#     print('-----------------------------------------------------------')

# start = True

# while start:
#     query = input('Enter Message:')
#     if query in ['quit','exit','bye']:
#         start = False
#         continue
#     search_results = get_google_search(query)
#     print(search_results[0]['link'])
#     res = try_get_content(search_results)
#     print(res)
