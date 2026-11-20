# Projeto Transformers

Mini-projeto de Inteligência Artificial que utiliza um **Transformer Encoder**, implementado em Python com PyTorch, para previsão de séries temporais meteorológicas do dataset **Jena Climate 2009–2016**.

O projeto possui dois objetivos:
- Prever o próximo token, correspondente a 1 hora.
- Prever os próximos 24 tokens, correspondentes a 24 horas.

Também foram realizados experimentos variando a quantidade de tokens de entrada, a dimensão do embedding, o número de cabeças de atenção e a quantidade de blocos codificadores.

## Instalação

Instale as dependências:

```bash
pip install -r requirements.txt
```

Baixe o [Jena Climate 2009–2016](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) e extraia o arquivo CSV para:

`data/jena_climate_2009_2016.csv`

## Execução

**Previsão de 1 hora:**

```bash
python main.py --horizon 1 --epochs 20
```

**Previsão de 24 horas:**

```bash
python main.py --horizon 24 --epochs 20
```

## Resultados

Os resultados dos seis experimentos estão disponíveis na pasta `results/`, incluindo:

- Métricas de avaliação (MAE e RMSE).
- Gráficos de perda de treinamento e validação.
- Comparações entre temperaturas reais e previstas.
- Mapas de atenção.
- Modelos treinados.

## Relatório

O relatório completo está disponível em:

[Relatório — Transformers](2026_Relatório_LuanDinizMazaroRodovalho.pdf)

**Autor:** Luan Diniz Mazaro Rodovalho  
**Instituição:** Universidade Federal de Uberlândia (UFU)
