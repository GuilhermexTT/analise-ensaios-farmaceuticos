import os
from serviços import dados, machine_learning

def main():
    caminho_csv = 'ensaios_clinicos.csv'
    caminho_modelo = 'modelo_ensaios.joblib'
    
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
        print("3. Sair")
        
        escolha = input("Escolha uma opção: ")
        
        if escolha == '1':
            dados.ingestao_e_limpeza(caminho_csv)
        elif escolha == '2':
            try:
                nova_dose = int(input("Digite a dose do medicamento (mg): "))
                nova_idade = int(input("Digite a idade do paciente: "))
                machine_learning.carregar_modelo_e_prever(nova_dose, nova_idade)
            except ValueError:
                print("Entrada inválida. Por favor, digite um número.")
        elif escolha == '3':
            print("Saindo da plataforma. Até mais!")
            break
        else:
            print("Opção inválida. Por favor, escolha 1, 2 ou 3.")

# Executa o programa
if __name__ == "__main__":
    main()