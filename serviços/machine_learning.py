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




# servicos/machine_learning.py
# (Código com todos os imports, train_test_split, DecisionTreeClassifier, etc.)
# ...

def carregar_modelo_e_prever(nova_dose, nova_idade, caminho_sustentabilidade):
    """
    Carrega o modelo salvo, faz a previsão clínica para o novo paciente
    e integra o contexto ambiental como alerta informativo.
    """
    try:

        # 1. Carrega o modelo treinado
        modelo_carregado = joblib.load('modelo_ensaios.joblib')
        
        # 2. Prepara os dados do novo paciente
        novo_paciente_data = np.array([[nova_dose, nova_idade]])
        
        # 3. Faz a previsão clínica
        previsao = modelo_carregado.predict(novo_paciente_data)
        
        print("\n--- Resultado da Análise Integrada ---")
        print(f"Previsão clínica do ensaio: {previsao[0]}")
        
        # --- INTEGRAÇÃO DO RELATÓRIO AMBIENTAL ---
        print("\n[CONTEXTO DE SUSTENTABILIDADE]")
        
        try:
            # 4. Lê os dados ambientais (separadamente)
            df_sust = pd.read_csv(caminho_sustentabilidade)
            
            # 5. Calcula e reporta a métrica ambiental (KPI)
            # Focamos na média de resíduos como exemplo de KPI de risco.
            media_residuos = df_sust['geracao_residuos_kg'].mean()
            
            print(f"Média de Resíduos dos lotes analisados: {media_residuos:.2f} kg/lote.")
            print("Status: Alerta gerado se os resíduos estiverem acima dos limites de conformidade.")
            print("Consulte a Opção 4 do Menu para ver a análise completa de risco ambiental.")
        
        except FileNotFoundError:
            print("[AVISO] Dados ambientais (CSV) não encontrados. Não foi possível fornecer o contexto de produção.")
        
    except FileNotFoundError:
        print("\n[ERRO FATAL] Arquivo 'modelo_ensaios.joblib' não encontrado. Por favor, treine o modelo primeiro (Opção 1).")
        
# ... (Treinar e salvar modelo deve vir antes ou depois, mas não é o que está sendo substituído) ...