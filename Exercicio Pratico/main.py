import sys
import os
import re
import pandas as pd
from nltk.stem import SnowballStemmer

caminho_dataset = sys.argv[1]

#lê as stopwords
with open(os.path.join(caminho_dataset, "stopwords.txt"), 'r', encoding='utf-8') as file:
        conteudo = file.read()  
        stopwords = conteudo.split("\n")

#lê os separadores
with open(os.path.join(caminho_dataset, "separadores.txt"), 'r', encoding='utf-8') as file:
        conteudo = file.read()  
        separadores = conteudo.split("\n")

docs = [] #lê os documentos
for doc in os.listdir(os.path.join(caminho_dataset, "documentos")): 
    caminho_doc = os.path.join(caminho_dataset, "documentos", doc)
    with open(caminho_doc, 'r', encoding='utf-8') as file:
        conteudo = file.read()  
        conteudo = re.findall(r'\w+', conteudo.lower(), flags=re.UNICODE) #normaliza os documentos
        conteudo_filtrado = [p for p in conteudo if p not in stopwords] #remove as stopwords
        docs.append(conteudo_filtrado)

#faz o stemming
stemmer = SnowballStemmer("portuguese")
stems = []
for i, doc in enumerate(docs):
    for j, palavra in enumerate(doc):
        stem = stemmer.stem(palavra)
        docs[i][j] = stemmer.stem(stem)
        if stem not in stems:
            stems.append(stem)

#calcula a matriz de frequências
matriz_frequencia = {}
matriz_frequencia[""] = stems
for i, doc in enumerate(docs):
    frequencias = []
    for stem in stems:
        frequencias.append(doc.count(stem))
    matriz_frequencia[f"documento {i+1}"] = frequencias


df = pd.DataFrame(matriz_frequencia)
print(df)