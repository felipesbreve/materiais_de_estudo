import requests
from flask import Flask, request, jsonify

import bs4 as bs
import urllib.request
import re
import nltk
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

import spacy

# Carregamento e pré-processamneto

dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()
dados_html = bs.BeautifulSoup(dados, 'lxml')
paragrafos = dados_html.find_all('p')

conteudo = ''
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower()
lista_sentencas = nltk.sent_tokenize(conteudo)

pln = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS

def preprocessamento(texto):
    #tirando URLs
    texto = re.sub(r"https://[A-Za-z0-9./]+", ' ', texto)
    #tirando espaços em braco
    texto = re.sub(r" +", ' ', texto)
    #lematização
    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)
        
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista

lista_sentencas_preprocessada = []
for i in range(len(lista_sentencas)):
    lista_sentencas_preprocessada.append(preprocessamento(lista_sentencas[i]))

# Boas-vindas

textos_boas_vindas_entrada = ('hey', 'olá', 'opa', 'oi', 'eae')
textos_boas_vindas_resrpostas = ('hey', 'olá', 'opa', 'oi', 'bem-vindo', 'como você está?')

def responder_saudacao(texto):
    for palavra in texto.split():
        if palavra.lower() in textos_boas_vindas_entrada:
            return random.choice(textos_boas_vindas_resrpostas)

#Função para o chatbot responder o usuário

def responder(texto_usuario):
    resposta_chatbot = ''
    lista_sentencas_preprocessada.append(texto_usuario)
    
    tfidf = TfidfVectorizer()
    palavras_vetorizadas = tfidf.fit_transform(lista_sentencas_preprocessada)
    
    similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
    
    indice_sentenca = similaridade.argsort()[0][-2]
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_encontrado = vetor_similar[-2]
    
    if(vetor_encontrado == 0):
        resposta_chatbot = resposta_chatbot + "Desculpe, mas não entendi!"
        return resposta_chatbot
    else:
        resposta_chatbot = resposta_chatbot + lista_sentencas[indice_sentenca]
        return resposta_chatbot



continuar = True
print('Olá, sou um chatbot e bou responder perguntas sobre Inteligência Artificial:')
while(continuar == True):
    texto_usuario = input()
    texto_usuario = texto_usuario.lower()
    if (texto_usuario != 'sair'):
        if (responder_saudacao(texto_usuario) != None):
            print('Chatbot: ' + responder_saudacao(texto_usuario))
        else:
            print('Chatbot: ')
            print(responder(preprocessamento(texto_usuario)))
            lista_sentencas_preprocessada.remove(preprocessamento(texto_usuario))
    else:
        continuar = False
        print('Chatbot: Até breve!')

# Criação da API Flask

app = Flask(__name__)

# Função para retornar as respostas

@app.route("/<string:txt>", methods = ["POST"])
def conversar(txt):
    resposta = ''
    texto_usuario = txt
    texto_usuario = texto_usuario.lower()
    if (responder_saudacao(texto_usuario) != None):
        resposta = responder_saudacao(texto_usuario)
    else:
        resposta = responder(preprocessamento(texto_usuario))
        lista_sentencas_preprocessada.remove(preprocessamento(texto_usuario))
    return jsonify({"texto_respondido": resposta})

# Iniciar a aplicação

app.run(port = 5000, debug = False)