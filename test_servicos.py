import unittest
import pandas as pd
import numpy as np
import os
import heapq # Necessário para o segundo teste

# Importa a função que será testada (necessária para o teste de dados)
# Corrija o import da sua função de limpeza se o nome do módulo for diferente
from serviços.dados import ingestao_e_limpeza 

# --- PARTE 1: TESTE DA INGESTÃO E LIMPEZA (Já Enviado) ---

# Cria um arquivo CSV de teste para que a função possa ler algo
caminho_teste = 'test_data.csv' 
conteudo_csv_teste = """id,idade,resultado
1,30,Eficaz
2,40,Ineficaz
3,,Eficaz 
4,50,Ineficaz
5,60,
"""
# Salva o arquivo CSV de teste no disco
with open(caminho_teste, 'w', newline='', encoding='utf-8') as f:
    f.write(conteudo_csv_teste)


class TestDadosPipeline(unittest.TestCase):
    
    def test_01_limpeza_de_dados_remove_nulos(self):
        """Testa se a funcao remove corretamente as linhas com valores nulos (NaN)."""
        df_limpo = ingestao_e_limpeza(caminho_teste)
        
        # 1. Verifica se o DataFrame foi retornado
        self.assertIsNotNone(df_limpo)
        
        # 2. Verifica se as linhas com erro (ID 3 e 5) foram removidas.
        self.assertEqual(len(df_limpo), 3, "O DataFrame deveria ter 3 linhas após a remoção de nulos.")
        
        # 3. Verifica se a contagem de nulos é zero
        self.assertEqual(df_limpo.isnull().sum().sum(), 0, "A contagem de nulos deveria ser zero.")

    def tearDown(self):
        """Remove o arquivo de teste após a execução de cada teste para limpar."""
        if os.path.exists(caminho_teste):
            os.remove(caminho_teste)
            
# --- PARTE 2: TESTE DA LÓGICA DA FILA DE PRIORIDADE (O que você perguntou) ---

class TestPrioridadeLogica(unittest.TestCase):
    
    def test_ordenacao_por_idade_decrescente(self):
        """
        Testa se o heap ordena corretamente os pacientes do mais velho 
        (maior prioridade) para o mais novo.
        """
        dados_brutos = [
            {'id_paciente': '101', 'idade_paciente': '30'},
            {'id_paciente': '102', 'idade_paciente': '60'},
            {'id_paciente': '103', 'idade_paciente': '45'},
        ]
        
        heap_temp = []
        
        # 1. CONSTRUÇÃO DO HEAP
        for row in dados_brutos:
            idade = int(row['idade_paciente'])
            prioridade = -idade # Inversão de prioridade
            heapq.heappush(heap_temp, (prioridade, row['id_paciente']))
            
        # 2. EXTRAÇÃO E VERIFICAÇÃO
        primeiro_saida = heapq.heappop(heap_temp) 
        segundo_saida = heapq.heappop(heap_temp) 
        
        # 3. AVALIAÇÃO (Asserts)
        self.assertEqual(primeiro_saida[1], '102', "O paciente mais velho (60 anos) deveria ter saído primeiro.")
        self.assertEqual(segundo_saida[1], '103', "O segundo paciente (45 anos) deveria ter saído em seguida.")

# --- EXECUÇÃO DOS TESTES ---
if __name__ == '__main__':
    # O unittest.main irá encontrar automaticamente AMBAS as classes de teste (TestDadosPipeline e TestPrioridadeLogica)
    unittest.main()

