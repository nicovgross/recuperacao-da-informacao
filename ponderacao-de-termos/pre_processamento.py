import os
import re
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
import math

def normalizar_documentos(caminho_dataset):
    #lê as stopwords
    with open(os.path.join(caminho_dataset, "stopwords.txt"), 'r', encoding='utf-8') as file:
            conteudo = file.read()  
            stopwords = conteudo.split("\n")

    docs = [] #lê os documentos
    for doc in os.listdir(os.path.join(caminho_dataset, "documentos")): 
        caminho_doc = os.path.join(caminho_dataset, "documentos", doc)
        with open(caminho_doc, 'r', encoding='utf-8') as file:
            conteudo = file.read()  
            conteudo = re.findall(r'\w+', conteudo.lower(), flags=re.UNICODE) #normaliza os documentos, já removendo os separadores
            conteudo_filtrado = [p for p in conteudo if p not in stopwords] #remove as stopwords
            docs.append(conteudo_filtrado)

    return docs

#faz o stemming de documentos
def stemming(docs):
    stemmer = SnowballStemmer("portuguese")
    stems = []
    for i, doc in enumerate(docs):
        for j, palavra in enumerate(doc):
            stem = stemmer.stem(palavra)
            docs[i][j] = stemmer.stem(stem)
            if stem not in stems:
                stems.append(stem)
    return stems

#faz o stemming de listas de termos
def stemming_lista(lista):
    stemmer = SnowballStemmer("portuguese")
    stems = []
    for palavra in lista:
        stem = stemmer.stem(palavra)
        if stem not in stems:
            stems.append(stem)
    return stems

#calcula a matriz de frequências
def calcula_matriz_frequencia(docs, stems):
    matriz_frequencia = {}
    matriz_frequencia[""] = stems
    for i, doc in enumerate(docs):
        frequencias = []
        for stem in stems:
            frequencias.append(doc.count(stem))
        matriz_frequencia[f"documento {i+1}"] = frequencias
    return pd.DataFrame(matriz_frequencia)

#calcula a matriz de frequências
def calcula_matriz_incidencia(docs, stems):
    matriz_incidencia = {}
    matriz_incidencia[""] = stems
    for i, doc in enumerate(docs):
        incidencias = []
        for stem in stems:
            if stem in doc:
                incidencias.append(1)
            else:
                incidencias.append(0)
        matriz_incidencia[f"documento {i+1}"] = incidencias
    return pd.DataFrame(matriz_incidencia)

def calcula_matriz_pesos(termos, matriz_frequencia):
    matriz_pesos = {}
    matriz_pesos[""] = termos
    N = matriz_frequencia.shape[1] 
    
    for j in range(N):
        matriz_pesos[f"documento {j+1}"] = []

    for termo in termos:
        n_i = sum(1 for j in range(N) if matriz_frequencia.loc[termo, f'documento {j+1}'] > 0)
        for j in range(N):
            freq = matriz_frequencia.loc[termo, f'documento {j+1}']
            if freq > 0 and n_i > 0:
                tf = 1 + math.log(freq, 2)
                idf = math.log(N / n_i, 2)
                peso = tf * idf
            else:
                peso = 0
            matriz_pesos[f"documento {j+1}"].append(peso)
    
    return pd.DataFrame(matriz_pesos)