# Aprendizagem Profunda 2025/2026 - Deteção de Texto Gerado por IA (LLMs)

Este repositório contém o código-fonte, dados exploratórios, e modelos desenvolvidos no âmbito da Unidade Curricular de Aprendizagem Profunda. O objetivo do projeto é a construção de um classificador multiclasse capaz de distinguir textos escritos por humanos de textos gerados por quatro famílias de Inteligência Artificial (OpenAI, Anthropic, Google, Meta).

## Identificação do Grupo

**Grupo 1 - Mestrado em Engenharia Informática (MEI)**

* Gabriel Torres (pg61523)
* Tomás Silva (pg60310)
* Diogo Costa (pg61513)
* João Costa (pg60269)
* André Carvalho (pg60237)

---

## Estrutura do Repositório

Abaixo apresenta-se a organização de diretórios e ficheiros do projeto:

```text
├── 📁 Apresentação Final/
├── 📁 data/                            # Diretório com os datasets limpos e de teste
├── 📁 modelos/                         # Pesos dos modelos treinados (.pt, .pkl) para a avaliação do docente
│
├── 📁 notebooks/                       # Cadernos Jupyter com exploração e treino documentado
│   ├── 02_model_training.ipynb
│   ├── 05_datasets_info.ipynb
│   ├── 08_gru.ipynb
│   ├── 09_lstm.ipynb
│   ├── 10_bert.ipynb
│   ├── 11_tfidf_mlp.ipynb
│   └── 12_ml_classic.ipynb
│
├── 📁 src/                             # Código fonte modular (scripts Python)
│   ├── data_utils.py                   # Funções de divisão estratificada e utilitários de dados
│   ├── numpy_models.py                 # Modelos implementados from scratch (Submissão 1)
│   ├── pytorch_models.py               # Arquiteturas de Deep Learning (RNN, GRU, LSTM, etc.)
│   ├── text_processing.py              # Pipeline de limpeza, regex e chunking de texto
│   └── train_utils.py                  # Funções de treino, avaliação e integração com MLflow
├── 📁 Subm1/
│   ├── subm1-g1-MEI-A.ipynb
│   ├── subm1-g1-MEI-A.csv
│   ├── subm1-g1-MEI-B.ipynb  
│   └── subm1-g1-MEI-B.csv  
├── 📁 Subm2/
│   ├── subm2-g1-MEI-A.ipynb
│   ├── subm2-g1-MEI-A.csv
│   ├── subm2-g1-MEI-B.ipynb  
│   └── subm2-g1-MEI-B.csv  
└── 📁 Subm3/
    ├── subm3-g1-MEI-A.ipynb
    ├── subm3-g1-MEI-A.csv
    ├── subm3-g1-MEI-B.ipynb  
    └── subm3-g1-MEI-B.csv  
```
