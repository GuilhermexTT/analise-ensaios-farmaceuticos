import csv
import heapq


def ordenar_pacientes_por_prioridade(caminho_csv):
    heap = []
    with open(caminho_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idade = row['idade_paciente']
            if idade:  
                
                heapq.heappush(heap, (-int(idade), row['id_paciente'], row))

    print("Pacientes em ordem de prioridade (idade decrescente):")
    while heap:
        _, id_paciente, dados = heapq.heappop(heap)
        print(f"Paciente {id_paciente}: idade {dados['idade_paciente']}, dose {dados['dose_medicamento']}, resultado {dados['resultado']}")