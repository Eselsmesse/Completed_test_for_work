import json
import random
import numpy as np
import pandas as pd
import string
import multiprocessing
import subprocess
import sys


# spacy

import spacy

from spacy.tokens import Doc, DocBin

from spacy.training.example import Example



# nltk

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


# gensim

from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import gensim

from gensim.models import KeyedVectors

from gensim.models import Word2Vec

from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.test.utils import datapath, get_tmpfile


#sklearn

from sklearn.metrics import accuracy_score

from tqdm import tqdm


def load_data(file):
    
    '''

    Description:

    Функция чтения данных
    

    input:

    file -> path to json file
    

    output:

    data -> json file
    '''
    

    with open(file, "r", encoding="utf-8") as f:

        data = json.load(f)

    return data


def save_data(file, data):

    '''

    Description:

    Функция сохранение данных
    

    input:

    file -> directory for save json file
    

    output:

    data -> json file
    '''
    

    with open (file, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4)
        

def create_train_valid(file):
    
    '''

    Description:

    функция принимает тренировочный файл с 

    обязательными колонками text, extracted_part, label

    и разделяет их на train и valid выборки
    

    input:

    file -> directory for save json file
    

    output:

    train_data, valid_data -> list
    '''
    

    train = pd.read_json(file)
        

    N = len(train)

    valid_idx = np.random.randint(N, size=N//5)

    train_idx = list(set(np.arange(N))-set(valid_idx))
        

    train_data = []

    valid_data = []

    for i in train_idx:

        train_data.append(

        [train.loc[i, 'text'], {'entities': [(train.loc[i, 'extracted_part']['answer_start'][0], 

                                                    train.loc[i, 'extracted_part']['answer_end'][0], 

                                                    train.loc[i, 'label'])]}]

    )
    

    for i in valid_idx:

        valid_data.append(

        [train.loc[i, 'text'], {'entities': [(train.loc[i, 'extracted_part']['answer_start'][0], 

                                                    train.loc[i, 'extracted_part']['answer_end'][0], 

                                                    train.loc[i, 'label'])]}]

    )
    

    print(f'Размер тренировачной выборки: {len(train_data)}')

    print(f'Размер валидационной выборки: {len(valid_data)}')

    return train_data, valid_data


def preprocess(text, STOPWORDS=stopwords.words('russian')):
    
    '''

    Describe:

    Функция вызывает связку функций preprocess_string и remove_stopwords

    из библиотеки spacy
    

    input:

    text -> str

    STOPWORDS -> list of stopwords
    

    output:

    sencente -> list of seqence
    '''
    

    text = text.lower()
    

    for sent in sent_tokenize(text):

        sentence = preprocess_string(remove_stopwords(sent, stopwords=STOPWORDS))


    return sentence


def training(model_name):

    '''

    Description:

    Обучает модель word_2_vec для нашей базы векторов
    

    Input:

    model_name -> str (path to save model)
    

    output:

    trained word2vec model 
    '''
    

    with open('data/work_vec_text.json', "r", encoding="utf-8") as f:

        texts = json.load(f)

    sentences = texts

    cores = multiprocessing.cpu_count()
    

    w2v_model = Word2Vec(min_count=5,

                         window=3,

                         vector_size=500,

                         sample=6e-5,

                         alpha=0.03,

                         min_alpha=0.0007,

                         negative=20,

                         workers=cores-1    

                        )

    w2v_model.build_vocab(texts)

    w2v_model.train(texts, 

                    total_examples=w2v_model.corpus_count,

                   epochs=30)

    w2v_model.save(f'word_vectors/{model_name}.model')

    w2v_model.wv.save_word2vec_format(f'word_vectors/word2vec_{model_name}.txt')
    

def gensim_similary(word, model_name):
    
    '''

    Description:

    Показывает ближайшие слова к заданному слову
    

    Input:

    word - str

    model_name -> str (path to txt file)
    

    output:

    list of closest word vectors to a given word
    '''


    model = KeyedVectors.load_word2vec_format(model_name,

                                             binary=False)

    results = model.most_similar(positive=[word])

    print(results)


def train_spacy(data, iterations):

    '''

    Description:

    Обучение модели nlp для решения задачи NER.
        

    input:

    data -> list

    iterations -> int
    

    output:

    nlp -> spacy.lang.ru.Russian
    '''
    

    TRAIN_DATA = data

    # идентификация базовой модели для русского языка

    nlp = spacy.blank("ru")

    # настройка пайплайна обучения

    if "ner" not in nlp.pipe_names:

        ner = nlp.add_pipe("ner", last=True)

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get('entities'):

            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):

        optimizer = nlp.begin_training()

        for itn in range(iterations):

            print("Starting iteration " + str(itn))

            random.shuffle(TRAIN_DATA)

            losses = {}

            for text, annotations in TRAIN_DATA:

                # создание образца

                doc = nlp.make_doc(text)

                nlp.update((text, annotations),

                           sgd=optimizer, 

                           losses=losses, 

                           drop=0.2

                          )
                

            print(losses)

    return nlp


def prepare_training(data):

    '''

    Description:

    Подготовка файлов в формат DocBin для проведение дальнейшего обучения
    

    input:

    data -> list
    

    output:

    db -> spacy.tokens._serialize.DocBin
    '''


    nlp = spacy.blank("ru")

    TRAIN_DATA = data
    

    db = DocBin()
    

    for text, annot in tqdm(TRAIN_DATA):

        doc = nlp.make_doc(text)

        ents = []

        for start, end, label in annot['entities']:

            span = doc.char_span(start, end, label=label, alignment_mode='contract')

            if span is None:

                print('Skipping entity')

            else:

                ents.append(span)

        doc.ents = ents

        db.add(doc)

    return db


def upsampling(data):

    '''

    Description:

    Функция увеличивает количество записей в 2 раза 

    переводя весь текст в нижний регистр.
    

    input:

    data -> list
    

    output:

    train_data -> list
    '''
    

    train_data = data
    

    print(f'Размер тренировочного датасета до upsampling: {len(train_data)}')
    

    for i in range(len(train_data)):

        train_data.append([train_data[0][0].lower(), 

                         train_data[0][1]])
        
    train_data = np.random.shuffle(train_data)
    print(f'Размер тренировочного датасета после upsampling: {len(train_data)}')

    return train_data


def train_spacy(data, iterations):

    '''

    Description:

    Обучение модели nlp для решения задачи NER.
        

    input:

    data -> list

    iterations -> int
    

    output:

    nlp -> spacy.lang.ru.Russian
    '''
    

    TRAIN_DATA = data

    # идентификация базовой модели для русского языка

    nlp = spacy.blank("ru")

    # настройка пайплайна обучения

    if "ner" not in nlp.pipe_names:

        ner = nlp.add_pipe("ner", last=True)

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get('entities'):

            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):

        optimizer = nlp.begin_training()

        for itn in range(iterations):

            print("Starting iteration " + str(itn))

            random.shuffle(TRAIN_DATA)

            losses = {}

            for text, annotations in TRAIN_DATA:

                # создание образца

                doc = nlp.make_doc(text)

                nlp.update((text, annotations),

                           sgd=optimizer, 

                           losses=losses, 

                           drop=0.2

                          )
                

            print(losses)

    return nlp


def extract_text(row):

    '''

    Description:

    Извлекает часть текст из колонки text по разметке из колонки extracted part
    

    input:

    row -> str
    

    output:

    text -> str
    '''


    text = row['text'][row['extracted_part']['entities'][0][0]:row['extracted_part']['entities'][0][1]]

    return text


def predict_text(row, model):

    '''

    Description:

    Функция возвращает именованную сущность 

    если таковая существует в иначе возвращает пустую строку
    

    input:

    row -> str

    model -> str (path to model dir)
    

    output:

    ent.text -> str
    '''

    nlp = spacy.load(model)
    

    if len(nlp(row).ents) > 0:

        return nlp(row).ents[0]

    else:

        return ''


def predict_test_text(row, model):
    
    '''
    Description:
    Функция возвращает именованную сущность 
    если таковая существует в иначе возвращает пустую строку
    
    input:
    row -> str
    model -> str (path to model dir)
        
    output:
    ent.text -> str
    '''
    
    nlp = spacy.load(model)
    

    if len(nlp(row).ents) > 0:

      ent = nlp(row).ents[0]

      return {"text": [ent.text],"answer_start": [ent.start_char],"answer_end": [ent.end_char]}

    else:

      return {"text": [''],"answer_start": [0],"answer_end": [0]}