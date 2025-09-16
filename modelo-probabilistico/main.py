import sys
import os
from nltk.stem import SnowballStemmer
from pre_processamento import *

caminho_dataset = sys.argv[1]

docs = normalizar_documentos(caminho_dataset)
stems = stemming(docs)

df_frequencias = calcula_matriz_frequencia(docs, stems)
matriz_pesos, df_pesos = calcula_matriz_pesos(stems, df_frequencias)

#print("------------------------- Matriz de pesos -------------------------")
#print(df_pesos, "\n")

#lê a consulta
with open(os.path.join(caminho_dataset, "consulta.txt"), 'r', encoding='utf-8') as file:
    conteudo = file.read()  
    consulta = conteudo.split()

k1 = input("Escolha um valor para K1: ")
try:
    k1 = float(k1)
except:
    print("--- Digite um valor válido! ---")
    sys.exit(1)
b = input("Escolha um valor para b: ")
try:
    b = float(b)
except:
    print("--- Digite um valor válido! ---")
    sys.exit(1)

print("\nConsulta: ", consulta)
consulta = stemming_lista(consulta)

# Calcula o ranking BM25
ranking = bm25_ranking(consulta, stems, df_frequencias, k1, b)

# Ordena por score desc
ranking = sorted(ranking, key=lambda x: x[1], reverse=True)

print("\nRanking dos documentos (BM25):")
for doc, score in ranking:
    print(f"{doc}: {score:.4f}")