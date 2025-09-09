import sys
import os
import pandas as pd
from nltk.stem import SnowballStemmer
from pre_processamento import *

caminho_dataset = sys.argv[1]

docs = normalizar_documentos(caminho_dataset)
stems = stemming(docs)

matriz_frequencia = calcula_matriz_frequencia(docs, stems)
matriz_frequencia.set_index("", inplace=True)
matriz_pesos = calcula_matriz_pesos(stems, matriz_frequencia)
matriz_pesos.set_index("", inplace=True)

print("------------------------- Matriz de pesos -------------------------")
print(matriz_pesos, "\n")

#caso nenhuma consulta seja especificada na linha de comando, lê a consulta do consulta.txt por padrão
if len(sys.argv) < 3:
    #lê a consulta
    with open(os.path.join(caminho_dataset, "consulta.txt"), 'r', encoding='utf-8') as file:
        conteudo = file.read()  
        consulta = conteudo.split()

    print("Consulta: ", consulta)
    consulta = stemming_lista(consulta) #faz o stemming da consulta

    resultado_consulta = []
    for i in range(matriz_frequencia.shape[1]):
        for j in range(len(consulta)):
            if matriz_frequencia.loc[consulta[j], f'documento {i+1}'] == 0:
                break
        else:
            resultado_consulta.append(f'documento {i+1}')

    print("Resultado da consulta: ", resultado_consulta)

else: #lê a consulta da linha de comando
    consulta = sys.argv[2].split()
    print("Consulta: ", consulta)
    operation = consulta[1].lower()
    if operation != "or" and operation != "and":
        print("Digite uma operação válida(and/or)")

    consulta = [p for p in consulta if p != operation]
    consulta = stemming_lista(consulta)
    
    consulta_binario = []
    for i in range(len(consulta)):
        binario = matriz_frequencia.loc[consulta[i]].to_list()
        consulta_binario.append(binario)
    
    # começa com o primeiro vetor
    resultado = consulta_binario[0]

    for vetor in consulta_binario[1:]:
        if operation == "and":
            resultado = [x & y for x, y in zip(resultado, vetor)]
        elif operation == "or":
            resultado = [x | y for x, y in zip(resultado, vetor)]

    # recuperar os documentos correspondentes
    docs_resultado = [
        f'documento {i+1}'
        for i, bit in enumerate(resultado)
        if bit == 1
    ]
    print("Resultado da consulta: ", docs_resultado)
