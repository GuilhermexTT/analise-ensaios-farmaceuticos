# Arquivo: servicos/dados.py
import pandas as pd
import numpy as np

def ingestao_e_limpeza(caminho_arquivo):
    """
    Função para ler um arquivo CSV, tratar valores nulos e retornar o DataFrame limpo.
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print("Dados ingeridos do arquivo com sucesso!")

        print("\nContagem de valores nulos antes da limpeza:")
        print(df.isnull().sum())
        df.dropna(inplace=True)
        print("\nLinhas com valores nulos foram removidas com sucesso!")

        return df

    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{caminho_arquivo}' não encontrado.")
        return None