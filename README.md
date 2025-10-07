# iguacu_ai
I2A2 - Agentes Inteligentes

Equipe: Iguaçu AI

Membros da equipe:
Bruno Ribeiro;
Jefferson L. G. Silva; 
José G. L. Filho

# 🤖 Autonomous Data Analysis Agent (EDA Agent) - iguacu_ai

Este projeto implementa um Agente Autônomo especializado em Análise Exploratória de Dados (EDA) genérica, utilizando Large Language Models (LLMs) com a técnica de Function Calling para interagir com um `DataFrame` de dados.

O agente é capaz de receber uma pergunta, planejar uma sequência de análises usando suas ferramentas programadas em Python, executar essas ferramentas e sintetizar os resultados em um formato compreensível para o usuário, incluindo gráficos.

## Estrutura do Projeto e Informações Obrigatórias

### 1. A Framework Escolhida

A solução principal utiliza:
* **LLM**: **OpenAI GPT-4o** (via API) como motor de raciocínio.
* **Agente**: **OpenAI Function Calling** (ou Tools) para a arquitetura do agente.
* **Interface**: **Gradio** para uma interface de chat web interativa, facilitando o carregamento de dados e a visualização imediata dos resultados.
* **Análise de Dados**: **Pandas**, **Matplotlib** e **Seaborn** para manipulação e visualização de dados.

### 2. Como a Solução Foi Estruturada

A solução foi estruturada de forma **modular** para separar a lógica de negócio (Análise de Dados), a inteligência do Agente (LLM Orchestration) e a interface de usuário:

1.  **Módulo `tools.py`**: Contém a definição do estado dos dados (`df`, `dataset_info`) e todas as **ferramentas de análise programadas** (ex: `analyze_distribution`, `analyze_correlation`). Essas ferramentas manipulam o `df` e retornam um resultado formatado em JSON.
2.  **Módulo `agent.py`**: Contém o **núcleo de raciocínio**. A classe `AgentMemory` gerencia o histórico e o contexto do dataset. A função `run_agent_core` implementa o loop de execução do agente, onde o LLM decide qual ferramenta chamar, executa o código Python correspondente e itera até gerar uma resposta final para o usuário.
3.  **Módulo `app.py`**: É o **ponto de entrada** da aplicação. Ele configura a interface Gradio, manipula o carregamento de arquivos CSV ou datasets do Kaggle, e conecta a entrada do usuário ao `run_agent_core`.

### 3. Pelo Menos 4 Perguntas com as Respectivas Respostas (e 1 gráfico)

Considerando um dataset genérico (ex: Titanic), o agente executa a seguinte sequência lógica:

| Pergunta | Tool(s) Acionada(s) | Resposta do Agente (Simulada) |
| :--- | :--- | :--- |
| **P1:** Qual a distribuição de idade (`Age`) dos passageiros? (Resultado Gráfico) | `analyze_distribution("Age")` | "A distribuição de idade revela que a maioria dos passageiros está na faixa dos 20-40 anos. O histograma mostra uma ligeira concentração de jovens adultos e outliers de idades mais avançadas. [Caminho para o gráfico gerado]" |
| **P2:** Qual a correlação entre a tarifa (`Fare`) e a idade (`Age`)? | `analyze_correlation()` | "A correlação entre Tarifa (`Fare`) e Idade (`Age`) é fraca (ex: `r = 0.12`), indicando que não há uma relação linear significativa entre a idade de um passageiro e o valor que ele pagou pela passagem." |
| **P3:** Quantos valores faltantes existem e quais colunas são mais afetadas? | `get_data_summary()` | "O resumo dos dados aponta que a coluna 'Cabin' possui 77% de valores faltantes. 'Age' também apresenta dados ausentes, com 19% de *missing values*. O agente precisa de estratégias de imputação para usar essas colunas em modelos." |
| **P4:** Existe diferença no preço médio da tarifa (`Fare`) entre as classes (`Pclass`)? | `compare_groups(column='Fare', group_by='Pclass')` | "Sim. A análise de comparação de grupos mostra que a classe 1 tem uma tarifa média significativamente maior do que as classes 2 e 3. Isso é claramente visível nos boxplots gerados pela ferramenta de comparação, que também apontam maior variância de preços na primeira classe." |

### 4. Uma Pergunta sobre as Conclusões do Agente e a Resposta

| Pergunta | Resposta do Agente (Simulada - Síntese Final) |
| :--- | :--- |
| **P:** Com base nas análises realizadas, quais são as principais conclusões e próximos passos recomendados? | **Conclusões Chave:** 1. **Qualidade dos Dados:** As colunas `Cabin` e `Age` requerem tratamento de *missing values*. 2. **Poder Preditivo:** A classe social (`Pclass`) é um forte fator discriminatório para a tarifa (`Fare`), o que sugere um alto poder preditivo para modelos. 3. **Distribuição:** A maior parte da população do dataset é composta por jovens adultos. **Próximos Passos:** 1. Imputar valores faltantes em `Age` usando a média ou mediana por grupo (`Pclass`). 2. Continuar a explorar a relação entre `Fare` e `Pclass` com outras variáveis categóricas. |

### 5. Os Códigos Fonte Gerados

Os códigos fonte gerados são os módulos Python apresentados em detalhes acima: **`app.py`**, **`agent.py`** e **`tools.py`**. O arquivo `requirements.txt` lista as dependências.

### 6. Um Link para Acessar seu Agente

Para obter um link permanente, a aplicação deve ser implantada usando o código do seu GitHub.

**Recomendação de Deploy: Hugging Face Spaces (Gradio)**

O caminho mais simples é usar um **Hugging Face Space** com o SDK Gradio, pois sua aplicação já utiliza Gradio.

1.  Crie um novo **Space** no Hugging Face (SDK Gradio).
2.  Conecte-o ao seu repositório GitHub (`https://github.com/jlgsilva/iguacu_ai`).
3.  Defina as seguintes variáveis de ambiente nos **Secrets** do Space (para ocultar chaves):
    * `OPENAI_API_KEY` (Obrigatório)
    * `KAGGLE_USERNAME` (Opcional, para carregar datasets do Kaggle)
    * `KAGGLE_KEY` (Opcional, para carregar datasets do Kaggle)

**Link de Acesso (Exemplo após Deploy):**
`[Link gerado pelo Hugging Face Spaces]` (O link será gerado após o deploy, ex: `https://huggingface.co/spaces/seu_usuario/iguacu_ai`)

### 7. Ocultar Chaves Utilizadas

Todas as chaves (`OPENAI_API_KEY`) são tratadas da seguinte forma:

1.  **Leitura Segura**: O código Python (em `app.py` e `agent.py`) lê a chave de API da variável de ambiente `OPENAI_API_KEY` usando a biblioteca `python-dotenv` ou o sistema de ambiente do Hugging Face/Streamlit.
2.  **Template**: O arquivo **`.env.example`** é fornecido como modelo, mas **não contém a chave real**.
3.  **Deploy**: Você deve configurar a chave real no painel de **Secrets/Variáveis de Ambiente** do seu serviço de hospedagem (Hugging Face Spaces, por exemplo).

## Como Executar a Aplicação Localmente

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/jlgsilva/iguacu_ai.git](https://github.com/jlgsilva/iguacu_ai.git)
    cd iguacu_ai
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate # Windows
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure a chave de API:**
    * Crie um arquivo chamado **`.env`** na raiz do projeto (use o `.env.example` como modelo).
    * Insira sua chave: `OPENAI_API_KEY="sk-..."`
5.  **Execute a aplicação:**
    ```bash
    python app.py
    ```
A interface Gradio será iniciada, e você poderá acessar o agente através do link local fornecido no seu terminal (ex: `http://127.0.0.1:7860`).
