# Importa as bibliotecas necessárias para o projeto
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- INÍCIO DA EXECUÇÃO DO PROJETO ---
if __name__ == "__main__":
    
    print("--- ROTINA DE INGESTÃO E LIMPEZA DE DADOS ---")
    
    # 1. Ingestão de Dados: Lê o arquivo CSV e carrega-o para um DataFrame (tabela)
    #    A tabela de dados 'df' agora contém todas as informações do arquivo.
    try:
        df = pd.read_csv('ensaios_clinicos.csv')
        print("Dados ingeridos do arquivo 'ensaios_clinicos.csv' com sucesso!")
    except FileNotFoundError:
        print("[ERRO] Arquivo 'ensaios_clinicos.csv' não encontrado. Certifique-se de que ele está na mesma pasta.")
        exit() # Encerra o programa se o arquivo não for encontrado

    # 2. Diagnóstico da Limpeza: Conta quantos valores nulos (NaN) existem em cada coluna
    print("\nContagem de valores nulos antes da limpeza:")
    print(df.isnull().sum())

    # 3. Limpeza de Dados: Remove todas as linhas que contêm valores nulos na tabela 'df'
    #    O parâmetro 'inplace=True' modifica o DataFrame 'df' diretamente.
    df.dropna(inplace=True)
    print("\nLinhas com valores nulos foram removidas com sucesso!")

    # 4. Verificação: Mostra a tabela de dados limpa e a nova contagem de nulos (que deve ser zero)
    print("\nTabela após a limpeza:")
    print(df)
    print("\nContagem de valores nulos após a limpeza:")
    print(df.isnull().sum())
    
    print("\n" + "="*50 + "\n")


    print("--- ROTINA DE CLASSIFICAÇÃO E PREVISÃO ---")

    # 5. Separação de Dados: Divide a tabela em entradas (X) e saída (y)
    #    X contém as colunas que o modelo usará para aprender.
    #    y contém a coluna com a resposta correta que o modelo tentará prever.
    X = df[['dose_medicamento', 'idade_paciente']]
    y = df['resultado']

    # 6. Divisão para Treino e Teste: Separa os dados de forma aleatória e reprodutível
    #    'test_size=0.3' reserva 30% dos dados para testar o modelo.
    #    'random_state=42' garante que a divisão seja a mesma sempre que o código for executado.
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Dados divididos em conjuntos de treino e teste.")

    # 7. Criação e Treinamento do Modelo
    #    Instancia um modelo de Árvore de Decisão para a nossa tarefa de classificação.
    modelo = DecisionTreeClassifier(random_state=42)
    
    #    O método .fit() é o "aprendizado". Ele ensina o modelo a encontrar padrões
    #    nos dados de treino.
    modelo.fit(X_treino, y_treino)
    print("Modelo de Árvore de Decisão treinado com sucesso!")
    
    # 8. Avaliação do Modelo: Mede a performance do modelo nos dados de teste
    #    O modelo faz previsões nos dados que ele NUNCA viu.
    previsoes_teste = modelo.predict(X_teste)

    #    Calcula a acurácia (porcentagem de acertos) comparando as previsões com as respostas corretas.
    acuracia = accuracy_score(y_teste, previsoes_teste)
    print(f"\nAcurácia do modelo em dados de teste: {acuracia*100:.2f}%")

    # 9. Previsão para um Novo Paciente (Demonstração)
    print("\n--- Previsão para um novo ensaio clínico ---")
    
    # Cria uma nova amostra de dados para prever.
    # É importante que o formato seja o mesmo do X_treino (uma 'tabela' com uma linha).
    nova_dose = 60
    nova_idade = 55
    novo_paciente_data = np.array([[nova_dose, nova_idade]])

    # Usa o modelo treinado para prever o resultado para este novo paciente
    previsao = modelo.predict(novo_paciente_data)
    
    print(f"Dados do novo paciente: Dose={nova_dose}mg, Idade={nova_idade} anos.")
    print(f"Previsão do resultado do ensaio: {previsao[0]}")
    
    print("\n" + "="*50 + "\n")
    print("Rotina de análise de ensaios concluída.")