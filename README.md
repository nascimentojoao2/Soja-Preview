# Soybean Price ML Dashboard

Projeto completo de Machine Learning com foco em **previsão de preço da soja** e **dashboard interativo em Streamlit**. Ele foi pensado para servir como peça de portfólio para vagas de **Machine Learning Engineer / AI Engineer**, especialmente com interesse em **AgTech**.

## Tecnologias escolhidas

- **Python**: linguagem principal
- **pandas / numpy**: manipulação de dados
- **scikit-learn**: treinamento e avaliação dos modelos
- **matplotlib**: gráficos
- **Streamlit**: dashboard web interativo
- **joblib**: persistência do modelo treinado

## Estrutura do projeto

```text
soybean-price-ml-dashboard/
├── app/
│   └── dashboard.py
├── data/
│   └── soybean_prices.csv
├── models/
│   ├── metrics.json
│   ├── processed_dataset.csv
│   └── soybean_model.pkl
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## O que o sistema faz

- carrega dados históricos de preço da soja
- cria features temporais e lag features
- treina dois modelos:
  - Linear Regression
  - Random Forest Regressor
- compara métricas de desempenho
- salva o melhor modelo
- abre um dashboard para prever preço futuro por data
- exibe gráficos de histórico e desempenho do modelo

## Como instalar

```bash
pip install -r requirements.txt
```

## Como treinar o modelo

```bash
cd soybean-price-ml-dashboard/src
python train_model.py
```

## Como executar o dashboard

```bash
streamlit run app/dashboard.py
```

## Dataset

O dataset incluso em `data/soybean_prices.csv` é **sintético**, gerado para deixar o projeto pronto para rodar imediatamente. Ele simula comportamento histórico de preços da soja e inclui variáveis auxiliares como:

- câmbio USD/BRL
- índice de chuva
- índice de oferta

## Melhorias futuras

- conectar com dados reais de mercado
- usar XGBoost ou LightGBM
- adicionar ingestão automática por API
- publicar o dashboard online
- prever mais de um horizonte temporal (1 dia, 7 dias, 30 dias)

## Ideia de apresentação no GitHub

Este projeto funciona muito bem como repositório de destaque no seu perfil, porque demonstra:

- Machine Learning aplicado ao agro
- organização de código
- pipeline completo
- interface web utilizável

## Licença

Uso educacional e de portfólio.
