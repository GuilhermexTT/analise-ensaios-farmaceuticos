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
    

def integrar_dados_para_analise(caminho_clinico, caminho_sustentabilidade):
    try:
        df_clinico = pd.read_csv(caminho_clinico)
        df_sust = pd.read_csv(caminho_sustentabilidade)
        
        print("\n--- Integração de Dados Clínicos e Ambientais ---")
        
        
        df_integrado = pd.merge(
            df_clinico, 
            df_sust, 
            on='nome_lote', 
            how='left'
        )
        
        
        print("\nDados Integrados (Exemplo de Rastreabilidade Lote-Paciente):")
        print(df_integrado[['id_paciente', 'idade_paciente', 'nome_lote', 'consumo_agua_litros']].head(5))

        return df_integrado
    
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo não encontrado: {e.filename}")
        return None
    except KeyError:
        print("[ERRO] Uma das colunas necessárias para o merge ('nome_lote') não foi encontrada em um dos arquivos.")
        return None
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro na integração dos dados: {e}")
        return None