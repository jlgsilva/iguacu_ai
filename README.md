# 🤖 iguacu_ai — Agente Autônomo de EDA (I2A2 - Agentes Inteligentes)

Equipe: Iguaçu AI  
Membros:  
- Bruno Ribeiro  
- Jefferson L. G. Silva  
- José G. L. Filho

Este projeto implementa um Agente Autônomo especializado em Análise Exploratória de Dados (EDA), utilizando LLM com Function Calling para orquestrar ferramentas de análise em um DataFrame. O agente recebe uma pergunta (ou um prompt autônomo), planeja, chama funções Python (tools) para executar análises e sintetiza os resultados para o usuário — incluindo gráficos salvos em `plots/`.

## 1. A Framework Escolhida
- LLM: OpenAI GPT-4o (via API)
- Agente: OpenAI Function Calling (Tools)
- Interface: Gradio (chat web interativo)
- Análise de Dados: Pandas, Matplotlib, Seaborn

## 2. Como a Solução Foi Estruturada
Arquitetura modular (mantendo a lógica funcional do notebook original):

- `tools.py`:  
  - Define o estado do dataset (`df`, `dataset_info`)  
  - Funções de análise (tools): `get_data_summary`, `analyze_distribution`, `analyze_correlation`, `detect_outliers`, `compare_groups`, além de placeholders `analyze_temporal_patterns` e `execute_custom_code`.  
  - Geração e salvamento de gráficos em `plots/`.
  - Manifesto das tools (`tools`) e mapeamento (`available_functions`).

- `agent.py`:  
  - `AgentMemory` para manter histórico, resultados de análises e conclusões.  
  - `run_agent_core`: laço principal do agente (LLM decide chamadas de tool, executa, agrega resultados e responde).  
  - `AUTONOMOUS_PROMPT_EXTENDED` e `autonomous_analysis` para disparar uma EDA completa automaticamente.

- `app.py`:  
  - Interface Gradio com upload, carregamento via Kaggle (opcional), execução da análise autônoma, chat interativo, e painéis de histórico/conclusões.

- `requirements.txt`: dependências.  
- `.env.example`: modelo de configuração de variáveis de ambiente.  

Observação: Os gráficos são salvos no diretório `plots/` e referenciados nas respostas como “Gráfico salvo em ...”. Em notebooks (Colab/Jupyter), também são exibidos via `display()`.

## 3. Perguntas e Respostas (exemplos)
Considerando um dataset genérico (ex.: Titanic ou Credit Card Fraud):

- P1: “Qual a distribuição de idade (Age) dos passageiros?”  
  Tool: `analyze_distribution("Age")`  
  Resposta:  
  “A distribuição de idade revela concentração entre 20–40 anos, com outliers em idades altas. Gráfico salvo em `plots/distribution_Age_*.png`.”

- P2: “Qual a correlação entre Fare e Age?”  
  Tool: `analyze_correlation()`  
  Resposta:  
  “A correlação entre Fare e Age é fraca (ex.: r ≈ 0.12), sugerindo ausência de relação linear forte.”

- P3: “Quantos valores faltantes existem e quais colunas são mais afetadas?”  
  Tool: `get_data_summary()`  
  Resposta:  
  “A coluna 'Cabin' possui forte presença de valores faltantes (ex.: ~77%). 'Age' também apresenta ~19% de missing.”

- P4: “Existe diferença no preço médio da tarifa (Fare) entre as classes (Pclass)?”  
  Tool: `compare_groups(column="Fare", group_by="Pclass")`  
  Resposta (com gráfico):  
  “Sim. A classe 1 apresenta média de tarifa significativamente maior do que classes 2 e 3. Boxplots e histogramas por grupo salvos em `plots/group_comparison_Fare_vs_Pclass_*.png`.”

## 4. Conclusão do Agente (exemplo)
Pergunta: “Com base nas análises, quais as principais conclusões e próximos passos?”
Resposta (síntese):  
- Qualidade dos Dados: ‘Cabin’ e ‘Age’ requerem imputação.  
- Poder Preditivo: ‘Pclass’ tem elevada influência sobre ‘Fare’.  
- Distribuição: População dominada por jovens adultos.  
Próximos Passos:  
- Imputar ‘Age’ por mediana estratificada por ‘Pclass’;  
- Explorar interações entre ‘Fare’, ‘Pclass’ e variáveis categóricas.

## 5. Códigos Fonte
- `app.py`, `agent.py`, `tools.py` (neste repositório).
- O notebook original (Google Colab) foi modularizado sem alterar a lógica funcional.

## 6. Link para Acessar o Agente
Recomendação de deploy: Hugging Face Spaces (Gradio).

Passos:
1. Crie um Space em https://huggingface.co/spaces (SDK Gradio).
2. Conecte ao GitHub: `https://github.com/jlgsilva/iguacu_ai`.
3. Configure os Secrets do Space:
   - `OPENAI_API_KEY` (obrigatório)
   - `KAGGLE_USERNAME` e `KAGGLE_KEY` (opcionais)
4. Defina o comando de execução (Space com Gradio geralmente detecta automaticamente):
   - Python: `app.py` como entrypoint.
5. O link do Space será algo como:  
   `https://huggingface.co/spaces/seu_usuario/iguacu_ai`

## 7. Ocultação de Chaves
- Não commit suas chaves no repositório.
- Use variáveis de ambiente:
  - Local: copie `.env.example` para `.env` e exporte no shell ou use `python-dotenv`.
  - Hugging Face Spaces: configure em “Settings > Secrets”.

---

## Como Rodar Localmente

1) Clone o repositório:
```bash
git clone https://github.com/jlgsilva/iguacu_ai.git
cd iguacu_ai

2) Crie e ative um ambiente virtual:

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

3) Instale as dependências:

pip install -r requirements.txt

4) Defina a variável de ambiente com sua chave OpenAI:
Método rápido (shell):

export OPENAI_API_KEY="sk-..."
# Windows PowerShell:
# $env:OPENAI_API_KEY="sk-..."

Ou crie um .env baseado em .env.example e exporte com sua ferramenta preferida.

5. Execute a aplicação:
python app.py

Acesse no navegador: http://127.0.0.1:7860 (ou a porta definida pela variável PORT).
