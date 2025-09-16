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
    dict_frequencias = {}
    dict_frequencias[""] = stems

    for i, doc in enumerate(docs):
        frequencias = []
        for stem in stems:
            frequencias.append(doc.count(stem))
        dict_frequencias[f"documento {i+1}"] = frequencias

    df_frequencias = pd.DataFrame(dict_frequencias)
    df_frequencias.set_index("", inplace=True)

    return df_frequencias


#calcula a matriz de incidência
def calcula_matriz_incidencia(docs, stems):
    dict_incidencia = {}
    dict_incidencia[""] = stems

    for i, doc in enumerate(docs):
        incidencias = []
        for stem in stems:
            if stem in doc:
                incidencias.append(1)
            else:
                incidencias.append(0)
        dict_incidencia[f"documento {i+1}"] = incidencias

    df_incidencias = pd.DataFrame(dict_incidencia)
    df_incidencias.set_index("", inplace=True)

    return df_incidencias

def calcula_matriz_pesos(termos, df_frequencia):
    dict_pesos = {}
    dict_pesos[""] = termos
    N = df_frequencia.shape[1]
    
    for j in range(N):
        dict_pesos[f"documento {j+1}"] = []

    for termo in termos:
        n_i = sum(1 for j in range(N) if df_frequencia.loc[termo, f'documento {j+1}'] > 0)
        for j in range(N):
            freq = df_frequencia.loc[termo, f'documento {j+1}']
            if freq > 0 and n_i > 0:
                tf = 1 + math.log(freq, 2)
                idf = math.log(N / n_i, 2)
                peso = tf * idf
            else:
                peso = 0
            dict_pesos[f"documento {j+1}"].append(peso)

    colunas_docs = [f"documento {j+1}" for j in range(N)]
    matriz_pesos = np.array([dict_pesos[col] for col in colunas_docs], dtype=float).T
    
    df_pesos = pd.DataFrame(dict_pesos)
    df_pesos.set_index("", inplace=True)
    
    return matriz_pesos, df_pesos #retorna a matriz de pesos em forma de matriz do numpy e dataframe fo pandas

def calcula_pesos_consulta(consulta, stems, df_frequencias):
    N = df_frequencias.shape[1]  # número de documentos
    pesos_consulta = []

    for termo in stems:
        f_iq = consulta.count(termo)
        n_i = sum(1 for j in range(N) if df_frequencias.loc[termo, f'documento {j+1}'] > 0)

        if f_iq > 0 and n_i > 0:
            tf = 1 + math.log(f_iq, 2)
            idf = math.log(N / n_i, 2)
            peso = tf * idf
        else:
            peso = 0
        pesos_consulta.append(peso)

    return np.array(pesos_consulta)

def calcula_similaridade(matriz_pesos, vetor_consulta):
    similaridades = []
    norma_q = np.linalg.norm(vetor_consulta)

    for doc in matriz_pesos.T:  # percorre colunas (cada documento)
        norma_d = np.linalg.norm(doc)
        if norma_d > 0 and norma_q > 0:
            sim = np.dot(doc, vetor_consulta) / (norma_d * norma_q)
        else:
            sim = 0
        similaridades.append(sim)

    return similaridades

def bm25_ranking(consulta, stems, df_frequencias, k1=1.0, b=0.75):
    N = df_frequencias.shape[1]  # número de documentos
    # tamanho de cada documento
    doc_lengths = df_frequencias.sum(axis=0).values
    avgdl = doc_lengths.mean()

    scores = [0.0 for _ in range(N)]

    for termo in consulta:
        if termo not in stems:
            continue

        n_t = sum(1 for j in range(N) if df_frequencias.loc[termo, f'documento {j+1}'] > 0)
        if n_t == 0:
            continue

        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1e-10)  # smoothing para evitar divisão por 0

        for j in range(N):
            f_tj = df_frequencias.loc[termo, f'documento {j+1}']
            if f_tj == 0:
                continue

            dl = doc_lengths[j]
            denom = f_tj + k1 * (1 - b + b * (dl / avgdl))
            score = idf * (f_tj * (k1 + 1)) / denom
            scores[j] += score

    return [(f"documento {i+1}", s) for i, s in enumerate(scores)]