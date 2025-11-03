---
title: "Processador de Notas Fiscais (NF-e)"
emoji: "📊"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "5.49.1"
app_file: "app.py"
pinned: false
license: "mit"
---

# 📊 Processador de Notas Fiscais (NF-e)

**Iguaçu AI — Grupo:**  
Bruno Ribeiro — bruno.ribsouza@gmail.com  
Jefferson Luiz Gonçalves Silva — j.lg11@yahoo.com.br  
José Gomes Lopes Filho — zefilho@msn.com

**Aplicação (Hugging Face Space):**  
🔗 [https://huggingface.co/spaces/jlgsilva/iguacu_ai_final](https://huggingface.co/spaces/jlgsilva/iguacu_ai_final)

**Licença:** MIT

---

## 🧭 Visão geral

Este aplicativo recebe um arquivo compactado (`.zip` ou `.7z`) contendo múltiplos XML de Notas Fiscais eletrônicas (NF-e), extrai e padroniza as principais informações de cada XML para compor um **CSV unificado** e, em seguida, executa uma **análise automatizada com suporte de linguagem grande (LLM — *Gemini 2.5 Flash*)**.  

A análise inclui **sumarização executiva**, **estatísticas temporais**, **detecção de anomalias**, **recomendações operacionais e comerciais**, além de **estimativas de emissões de CO₂** associadas aos insumos adquiridos.  
O app também gera **visualizações (gráficos)** e oferece um **chat interativo** alimentado pela mesma LLM para perguntas *ad-hoc* sobre os dados.

---

## 🎯 Temas escolhidos

1. **Extração de Dados**  
2. **Classificação, categorização e customização por ramo de atividade**  
3. **Ferramentas gerenciais**

### 📌 Justificativa

- **Extração de Dados:** O núcleo do app é a extração e padronização de informações a partir de múltiplos XML de NF-e, consolidando tudo em um único CSV.  
- **Classificação e Categorização:** A aplicação classifica itens por palavras-chave nas descrições, permitindo análises por categoria e estimativas de emissões.  
- **Ferramentas Gerenciais:** O app entrega dashboards, análises e recomendações automáticas úteis para gestão de compras e controle de fornecedores.

---

## 👥 Público-alvo

- Auditorias e equipes de controle interno (TCs, CGUs, auditores independentes).  
- Departamentos de compras e suprimentos (públicos e privados).  
- Gestores de contratos e compliance.  
- Analistas financeiros e contábeis.  
- Pesquisadores e consultorias em gastos públicos.

---

## ⚙️ Funcionalidades principais

### 🧩 Etapas do processamento

1. **Upload** do arquivo `.zip` ou `.7z` via interface Gradio.  
2. **Extração recursiva** de todos os arquivos XML (incluindo pastas e subpastas).  
3. **Leitura e interpretação** de cada XML NF-e (estrutura `infNFe`).  
4. **Extração dos principais campos:**
   - **Metadados:** chave, número, data, natureza da operação, modelo, série, tipo.  
   - **Emitente e destinatário:** CNPJ, nome.  
   - **Totais:** valor total (`vNF`).  
   - **Itens:** código, descrição, NCM, CFOP, unidade, quantidade, valor.  
5. **Criação do CSV unificado** (`notas_fiscais.csv`) com todas as notas.  
6. **Análises automáticas:**
   - Estatísticas temporais (médias, totais, variação mensal).  
   - Ranking de fornecedores e categorias de produtos.  
   - Gráficos automáticos (gastos, top itens, emissões).  
   - Estimativa de emissões de CO₂ com base nas categorias de produtos.  
7. **Análise textual inteligente (LLM Gemini 2.5 Flash):**
   - Síntese executiva e recomendações gerenciais.  
   - Identificação de anomalias e oportunidades de economia.  
8. **Interface interativa:**
   - Download do CSV consolidado.  
   - Visualização dos gráficos.  
   - Chat inteligente para consultas sobre os dados.

---

## 📄 Estrutura do CSV unificado

| Coluna | Descrição |
|:-------|:-----------|
| chave | Id da NF-e |
| numero | Número da nota |
| data_emissao | Data/hora de emissão |
| natureza_operacao | Natureza da operação |
| modelo | Modelo do documento |
| serie | Série da nota |
| tipo_operacao | Tipo (entrada/saída) |
| emitente_cnpj | CNPJ do emitente |
| emitente_nome | Nome do emitente |
| destinatario_cnpj | CNPJ do destinatário |
| destinatario_nome | Nome do destinatário |
| valor_nf | Valor total da NF |
| itens | JSON com lista de itens e seus atributos |

---

## 📊 Saídas geradas

- **CSV:** `notas_fiscais.csv` consolidado.  
- **Gráficos automáticos:**
  - `monthly_spending.png` — Gastos mensais  
  - `top_items.png` — Top 10 itens  
  - `co2_emissions.png` — Emissões de CO₂ mensais  
- **Relatório analítico gerado pela LLM Gemini.**  
- **Chat interativo** com análise contextual.

---

## 🌱 Metodologia de estimativa de CO₂

As descrições dos itens são classificadas automaticamente em categorias, cada uma com um fator médio de emissão (kg CO₂ / R$):

| Categoria | Fator (kg CO₂/R$) |
|:-----------|:------------------:|
| Alimentos | 0.5 |
| Eletrônicos | 1.2 |
| Construção | 0.8 |
| Limpeza | 0.3 |
| Vestuário | 0.6 |
| Móveis | 0.7 |
| Outros | 0.5 |

Os valores são usados para estimar emissões mensais e totais, apresentadas em gráficos e relatórios.

---

## 💻 Como executar localmente

```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Definir chave da API Gemini
export GEMINI_API_KEY="sua_chave_aqui"

# 4. Executar
python app.py

---

## 📦 Exemplo de `requirements.txt`

gradio==5.49.1
py7zr
pandas
numpy
matplotlib
google-genai


---

## ⚠️ Boas práticas e limitações

- Verifique a conformidade e privacidade das NF-e utilizadas.  
- Datas com formatos não padronizados podem exigir revisão.  
- Os fatores de emissão são aproximados e servem para análises exploratórias.  
- A qualidade das respostas da LLM depende do contexto e do resumo de dados enviados.  
- Em grandes volumes de XML, recomenda-se pré-processar por lotes.

---

## 🚀 Melhorias futuras

- Exportação opcional de JSONs individuais.  
- Classificação automática de categorias via modelo treinável.  
- Painel interativo (*dash*) com filtros por fornecedor, NCM e valor.  
- Armazenamento histórico e suporte a eventos de NF-e.

---

## 📬 Contato e créditos

**Equipe:** Iguaçu AI  
**Integrantes:**  
Bruno Ribeiro — bruno.ribsouza@gmail.com  
Jefferson Luiz Gonçalves Silva — j.lg11@yahoo.com.br  
José Gomes Lopes Filho — zefilho@msn.com  

**Space:**  
🔗 [https://huggingface.co/spaces/jlgsilva/iguacu_ai_final](https://huggingface.co/spaces/jlgsilva/iguacu_ai_final)

---

## 📚 Referências

- **Layout oficial NF-e:** Portal Nacional da NF-e  
- **LLM:** Gemini 2.5 Flash (Google GenAI)  
- **Fatores de emissão:** referências DEFRA / IPCC (valores médios)

---
