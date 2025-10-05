import os
from serviços import dados, machine_learning, heapq_servicos 

def main():
    caminho_csv = 'ensaios_clinicos.csv'
    caminho_modelo = 'modelo_ensaios.joblib'
    caminho_csv_sustentabilidade = 'relatorio_sustentabilidade.csv'
    
    # Verifica se o modelo já foi treinado e salvo.
    if not os.path.exists(caminho_modelo):
        print("Modelo de classificação não encontrado. Iniciando o treinamento...")
        df = dados.ingestao_e_limpeza(caminho_csv)
        if df is not None:
            machine_learning.treinar_e_salvar_modelo(df)
            
    while True:
        print("\n--- MENU DA PLATAFORMA ---")
        print("1. Ingestão e Limpeza de Dados")
        print("2. Fazer Previsão de Novo Paciente")
        print("3. Ordenar Pacientes por Idade (Fila de Prioridade)")
        print("4. Integração de Dados")
        print("5. Sair")
        
        escolha = input("Escolha uma opção: ")
        
        if escolha == '1':
            dados.ingestao_e_limpeza(caminho_csv)

        elif escolha == '2':
            try:
                nova_dose = int(input("Digite a dose do medicamento (mg): "))
                nova_idade = int(input("Digite a idade do paciente: "))
        
        # AQUI ESTÁ A CHAVE: PASSAR O CAMINHO DO ARQUIVO AMBIENTAL
                machine_learning.carregar_modelo_e_prever(
                nova_dose, 
                nova_idade, 
                caminho_csv_sustentabilidade 
            )
            except ValueError:
                print("Entrada inválida. Por favor, digite um número.")
                
        elif escolha == '3':
            print("Ordenando pacientes.")
            caminho_csv = 'ensaios_clinicos.csv'
            heapq_servicos.ordenar_pacientes_por_prioridade(caminho_csv)
        elif escolha == "4":
            print("Iniciando a integração de dados.")
            dados_integrados = dados.integrar_dados_para_analise(caminho_csv, caminho_csv_sustentabilidade) 
            if dados_integrados is not None:
                print(f"Integração concluída! Total de linhas no DataFrame: {len(dados_integrados)}")
        elif escolha == "5":
            print("Encerrando o processo.")
            break


# Executa o programa
if __name__ == "__main__":
    main()