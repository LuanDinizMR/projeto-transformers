# Projeto Transformers

Mini-projeto de Inteligência Artificial para previsão de séries temporais meteorológicas usando **Transformer Encoder (PyTorch)** e o dataset **Jena Climate 2009–2016**.

## Como executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. [Baixe o Jena Climate](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) e extraia o CSV para `data/jena_climate_2009_2016.csv`.

3. Execute os experimentos:
   ```bash
   # Previsão da próxima hora
   python main.py --horizon 1 --epochs 20

   # Previsão das próximas 24 horas
   python main.py --horizon 24 --epochs 20
   ```

Os gráficos e as métricas são salvos na pasta `results/`. Os resultados dos experimentos realizados estão disponíveis em `results.zip`.

O relatório do trabalho está disponível em **`2026_Relatório_LuanDinizMazaroRodovalho.pdf`**.
