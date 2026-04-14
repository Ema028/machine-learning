# 🤖 Machine Learning
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-blueviolet)
![Status](https://img.shields.io/badge/status-active-success)

Este repositório é um guarda-chuva com meus experimentos, provas de conceito e pipelines de dados. O objetivo é implementar na prática os principais algoritmos de IA clássica, analisando o comportamento de cada modelo.

## 🛠️ Padrão do Repositório

* **Módulo Customizado (`/utils`):** O repositório conta com uma biblioteca própria (`pre_processing.py`) com uma classe `Dataframe` com diversas funções auxiliares de tratamento e visualização para reaproveitamento de código.
* **Métricas e Avaliação:** Todos os modelos possuem análise de métricas pertinentes ao seu tipo (Acurácia, Matriz de Confusão, AUC ROC, Relatório de Classificação, RMSE, R², etc.).
* **Visualização de Dados:** Cada projeto possui uma subpasta `/graficos` contendo os pngs visuais gerados (linhas de regressão, dispersão de clusters, etc.).
* **Storytelling e Insights:** Os códigos são comentados com análises críticas, justificativas matemáticas e comparações entre diferentes abordagens.

---

## ⚙️ Configurando o Ambiente

Este projeto utiliza um ambiente virtual para gerenciar dependências de forma isolada.

### Passo 1: Crie o Ambiente Virtual
Crie um ambiente virtual na pasta do projeto:
 
```bash
python -m venv .venv
```

### Passo 2: Instale as Dependências
Instale as dependências do projeto listadas no arquivo `requirements.txt`. Escolha o comando abaixo de acordo com o seu sistema operacional:

**No Linux:**
```bash
.venv/bin/pip install -r requirements.txt
```

**No Windows:**
```bash
.venv\Scripts\pip install -r requirements.txt
```

---

## 📂 Projetos e Algoritmos

### 📈 Modelos de Regressão
* **[Regressão Linear](./linear-regression)**
  * **Objetivo:** Previsão do valor de aluguéis de alto padrão baseado em metragem, taxa de condomínio, entre outros.
  * **Destaques:** Testes comparativos entre Regressão Linear Simples e Múltipla.
* **[Regressão Polinomial](./polynomial-regression)**
  * **Objetivo:** Evolução do modelo de aluguéis capturando relações não-lineares.
  * **Destaques:** Comparação direta com a Regressão Linear. Testes com graus polinomiais (Degree 2 e 4) isolando features, além da aplicação de Regularização **Lasso** no modelo múltiplo completo.

### 🎯 Modelos de Classificação
* **[Árvore de Decisão](./decision-tree)**
  * **Objetivo:** Prever se um cliente terá uma pontuação de crédito alta.
  * **Destaques:** Utilização de **Cross-Validation**. Análise comparativa entre uma árvore completa e uma árvore reduzida (pruning), com a reduzida apresentando melhor desempenho de generalização. Contém comparativos de performance com o modelo **Naive Bayes**.
* **[Naive Bayes](./naive-bayes)**
  * **Objetivo:** Previsão de pontuação de crédito alta (mesma base da Árvore de Decisão).
  * **Destaques:** Validação com **Cross-Validation** e análise probabilística.
* **[Random Forest](./random-forest)**
  * **Objetivo:** Análise e previsão da qualidade de vinhos.
  * **Destaques:** Avaliação de modelo completo vs. reduzido (**Feature Selection**). Otimização de hiperparâmetros utilizando **Random Search**. Melhoria drástica de performance na etapa final aplicando técnicas de **Binning** e balanceamento com **SMOTE**.
* **[Detecção de Fumaça em Sensores IoT](./smoke_detection)**
  * **Objetivo:** Prever a ocorrência de incêndios utilizando dados de sensores ambientais.
  * **Destaques:** Identificação e correção de **Data Leakage** (isolamento de variáveis temporais). Implementação de Regressão Logística como **baseline**. Otimização de hardware orientada por **Feature Importance**, reduzindo a placa IoT de 12 para apenas 3 sensores (TVOC, Etanol e Pressão), mantendo performance sob **Cross-Validation** rigorosa.

### 🧩 Modelos de Clusterização (Não-Supervisionados)
* **[K-Means: Pinguins](./penguins-clustering-kmeans)**
  * **Objetivo:** Clusterização de espécies de pinguins.
  * **Destaques:** Definição de clusters utilizando os métodos **Elbow** e **Silhouette**. Validação externa dos agrupamentos utilizando a métrica **adjusted_rand_score**.
* **[K-Means: Segmentação de Clientes](./segmentacao-clientes-kmeans)**
  * **Objetivo:** Segmentação da base de clientes de um shopping.
  * **Destaques:** Cruzamento final dos clusters com dados demográficos (gênero e idade). Análise profunda do perfil de cada grupo com foco em **estratégia comercial e negócios**.

---

## ⏳ Em Breve (Na Fila de Desenvolvimento)
* **XGBoost:** Implementação de árvores otimizadas com gradiente.
* **Support Vector Machines (SVM):** Classificação de margem máxima.
* **Pipelines Completos (End-to-End):** 
  * Projeto focando em redução de dimensionalidade com **PCA**.
  * +2 Projetos práticos completos, com a seleção do algoritmo ditada estritamente pela natureza da base de dados e do problema de negócio.