import sys
import os
from nltk.stem import SnowballStemmer
from pre_processamento import *

caminho_dataset = sys.argv[1]

docs = normalizar_documentos(caminho_dataset)
stems = stemming(docs)

df_frequencias = calcula_matriz_frequencia(docs, stems)
matriz_pesos, df_pesos = calcula_matriz_pesos(stems, df_frequencias)

print("------------------------- Matriz de pesos -------------------------")
print(df_pesos, "\n")

#lê a consulta
with open(os.path.join(caminho_dataset, "consulta.txt"), 'r', encoding='utf-8') as file:
    conteudo = file.read()  
    consulta = conteudo.split()

print("Consulta: ", consulta)
consulta = stemming_lista(consulta) #faz o stemming da consulta

vetor_consulta = calcula_pesos_consulta(consulta, stems, df_frequencias)

similaridades = calcula_similaridade(matriz_pesos, vetor_consulta)
#faz o ranking dos documentos 
ranking = sorted(
    [(f"documento {i+1}", sim) for i, sim in enumerate(similaridades)],
    key=lambda x: x[1],
    reverse=True
)

print("\nRanking dos documentos:")
for doc, score in ranking:
    print(f"{doc}: {score:.4f}")