import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def treinar_e_salvar_modelo(df):
    """
    Função para treinar o modelo de classificação e salvá-lo em um arquivo.
    """
    print("\n--- Treinando o Modelo de Classificação ---")
    
    # Separação dos dados de entrada (X) e saída (y)
    X = df[['dose_medicamento', 'idade_paciente']]
    y = df['resultado']

    # Divisão para treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criação e treinamento do modelo
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_treino, y_treino)
    
    print("Modelo de Árvore de Decisão treinado com sucesso!")

    # Avaliação do modelo
    previsoes_teste = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes_teste)
    print(f"Acurácia do modelo em dados de teste: {acuracia*100:.2f}%")
    
    # Salva o modelo treinado em um arquivo
    joblib.dump(modelo, 'modelo_ensaios.joblib')
    print("Modelo salvo como 'modelo_ensaios.joblib'.")

def carregar_modelo_e_prever(nova_dose, nova_idade):
    """
    Função para carregar o modelo salvo e fazer uma previsão.
    """
    try:
        # Carrega o modelo do arquivo
        modelo_carregado = joblib.load('modelo_ensaios.joblib')
        print("\nModelo de classificação carregado com sucesso!")
        
        # Prepara os dados do novo paciente no formato correto
        novo_paciente_data = np.array([[nova_dose, nova_idade]])
        
        # Faz a previsão
        previsao = modelo_carregado.predict(novo_paciente_data)
        
        print(f"\n--- Previsão para um novo ensaio clínico ---")
        print(f"Dados do novo paciente: Dose={nova_dose}mg, Idade={nova_idade} anos.")
        print(f"Previsão do resultado do ensaio: {previsao[0]}")
        
    except FileNotFoundError:
        print("\n[ERRO] Arquivo 'modelo_ensaios.joblib' não encontrado. Treine o modelo primeiro.")