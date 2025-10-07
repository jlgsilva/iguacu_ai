# agent.py
import os
import json
from openai import OpenAI

from tools import (
    df, dataset_info, tools, available_functions,
    get_data_summary, analyze_distribution, analyze_correlation,
    detect_outliers, compare_groups, analyze_temporal_patterns,
    execute_custom_code
)

# Memória do agente
class AgentMemory:
    def __init__(self):
        self.conversation_history = []
        self.analysis_results = {}
        self.conclusions = []
        self.dataset_context = {}

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def add_analysis(self, analysis_type, result):
        self.analysis_results[analysis_type] = result

    def add_conclusion(self, conclusion):
        if conclusion not in self.conclusions:
            self.conclusions.append(conclusion)

    def set_dataset_context(self, context):
        self.dataset_context = context

    def get_context(self):
        context = ""
        if self.dataset_context:
            context += "Informações do Dataset:\n"
            context += f"- Arquivo: {self.dataset_context.get('filename', 'N/A')}\n"
            context += f"- Dimensões: {self.dataset_context.get('shape', 'N/A')}\n"
            context += f"- Colunas: {', '.join(self.dataset_context.get('columns', []))}\n\n"
        if self.analysis_results:
            context += "Análises realizadas:\n"
            for analysis_type in list(self.analysis_results.keys())[-5:]:
                context += f"- {analysis_type}\n"
        if self.conclusions:
            context += "\nConclusões anteriores:\n"
            for i, conclusion in enumerate(self.conclusions, 1):
                context += f"{i}. {conclusion}\n"
        return context

    def get_full_history(self):
        return self.conversation_history[-10:]

    def get_initial_history(self):
        return self.conversation_history

    def reset(self):
        self.conversation_history = []
        self.analysis_results = {}
        self.conclusions = []

memory = AgentMemory()

def identify_best_column_for_auto_plot():
    global df, dataset_info
    if df is None or df.empty:
        return None, None
    numeric_cols = dataset_info.get('numeric_columns', [])
    categorical_cols = dataset_info.get('categorical_cols', [])
    if numeric_cols:
        try:
            std_devs = df[numeric_cols].std().sort_values(ascending=False)
            if not std_devs.empty:
                best_col = std_devs.index[0]
                return best_col, "numérica com maior variância"
        except:
            pass
    if categorical_cols:
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                return col, "categórica ideal para distribuição"
        if categorical_cols:
            return categorical_cols[0], "primeira coluna categórica"
    if dataset_info.get('columns'):
        if numeric_cols:
            return numeric_cols[0], "primeira coluna numérica"
        else:
            return dataset_info['columns'][0], "primeira coluna disponível"
    return None, None

def run_agent_core(user_query, history=None):
    from tools import df, dataset_info, tools, available_functions  # garantir estado atualizado
    if df is None:
        return "⚠️ Nenhum dataset carregado. Por favor, carregue um arquivo CSV primeiro."
    memory.set_dataset_context(dataset_info)
    system_message = f"""Você é um agente especializado em Análise Exploratória de Dados (EDA) genérico e adaptável, focado em fornecer insights detalhados e conclusões.

Dataset atual:
- Arquivo: {dataset_info.get('filename', 'desconhecido')}
- Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas
- Colunas numéricas ({len(dataset_info.get('numeric_columns', []))}): {', '.join(dataset_info.get('numeric_columns', [])[:10])}{'...' if len(dataset_info.get('numeric_columns', [])) > 10 else ''}
- Colunas categóricas ({len(dataset_info.get('categorical_columns', []))}): {', '.join(dataset_info.get('categorical_columns', [])[:10])}{'...' if len(dataset_info.get('categorical_columns', [])) > 10 else ''}

Você tem acesso a ferramentas de análise (get_data_summary, analyze_distribution, analyze_correlation, etc.).

Contexto das análises anteriores:
{memory.get_context()}

IMPORTANTE:
- Para a análise inicial ou se for solicitado um resumo, use get_data_summary() primeiro.
- Use as tools de forma sequencial e lógica.
- Adapte suas análises ao tipo de dados.
- Sintetize as informações em conclusões acionáveis.
- Seja proativo em sugerir análises relevantes.
- Responda em português de forma clara e estruturada com markdown.
"""
    if history is not None:
        memory.add_message("user", user_query)
        messages = [{"role": "system", "content": system_message}] + memory.get_full_history()
    else:
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}]
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    max_iterations = 10
    iteration = 0
    final_response_content = None
    while iteration < max_iterations:
        iteration += 1
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        if not getattr(response_message, "tool_calls", None):
            final_response_content = response_message.content
            if final_response_content and any(k in final_response_content.lower() for k in ["conclusões", "insights", "resumo dos achados"]):
                memory.add_conclusion(final_response_content)
            memory.add_message("assistant", final_response_content)
            if history is not None:
                history.append([messages[-1]["content"] if messages else "", final_response_content])
                return history
            return final_response_content
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            memory.add_analysis(function_name, function_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response
            })
    error_message = "⚠️ Número máximo de iterações atingido"
    if history is not None:
        history.append([messages[-1]["content"] if messages else "", error_message])
        return history
    return error_message

AUTONOMOUS_PROMPT_EXTENDED = """
Realize uma Análise Exploratória de Dados (EDA) completa e detalhada deste dataset...
## 1. Descrição e Estrutura dos Dados (Obrigatório)
1. Resumo Inicial: Use a tool get_data_summary()
2. Identificação de Tipos
3. Valores Faltantes
4. Duplicatas
## 2. Análise Univariada e Distribuição (Inclui Gráfico Automático)
5. Distribuição Gráfica: GERE IMEDIATAMENTE UM GRÁFICO com analyze_distribution
6-8. Medidas de tendência e variabilidade; frequências categóricas
## 3. Detecção de Anomalias (Outliers)
9-10. detect_outliers em pelo menos duas colunas e recomendações
## 4. Relações e Padrões (Multivariada)
11-14. analyze_correlation (gráfico), top correlações, compare_groups
## 5. Análises Adicionais
15-24. Temporal (se aplicável), homogeneidade, features, balanceamento, cardinalidade, bi-variadas, quartis, skew/kurtosis, próximos passos, clusters
## 6. Conclusões Finais
25. Síntese dos achados.
"""

def autonomous_analysis():
    from tools import df  # estado atualizado
    if df is None:
        return [], "❌ Nenhum dataset carregado para análise autônoma."
    memory.reset()
    col_to_plot, reason = identify_best_column_for_auto_plot()
    if not col_to_plot:
        initial_response = "❌ Erro: Não foi possível identificar colunas para análise."
        return [], initial_response
    initial_analysis_prompt = f"""
{AUTONOMOUS_PROMPT_EXTENDED}

EXECUÇÃO INICIAL FORÇADA:
- Use get_data_summary() primeiro.
- Em seguida execute analyze_distribution(column='{col_to_plot}').
- Continue com o plano de análise.
"""
    final_response = run_agent_core(initial_analysis_prompt, history=None)
    analysis_history = memory.get_initial_history()
    formatted_history = []
    for item in analysis_history:
        if item["role"] == "assistant":
            if formatted_history and formatted_history[-1][1] is None:
                formatted_history[-1][1] = item["content"]
            else:
                formatted_history.append(["**Análise Autônoma**", item["content"]])
    if formatted_history and formatted_history[-1][1] is not None:
        formatted_history.append([None, "👋 **Análise Autônoma Concluída!**\n\nAgora, sinta-se à vontade para fazer perguntas detalhadas sobre os dados."])
    return formatted_history, final_response
