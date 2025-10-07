# tools.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown, Image  # no-op fora de notebook
# Em ambientes não-notebook, display não é necessário; o arquivo é salvo em plots/.

# Estado global do dataset
df = None
dataset_info = {}

# ----------------------------
# Utilitários de carregamento
# ----------------------------

def update_dataset_info(csv_path):
    global df, dataset_info
    dataset_info = {
        "filename": os.path.basename(csv_path),
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "missing_summary": df.isnull().sum().to_dict()
    }
    info_text = f"""✅ Dataset carregado com sucesso!
📊 Shape: {df.shape[0]} linhas × {df.shape[1]} colunas
📋 Colunas: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
🔢 Colunas numéricas: {len(dataset_info['numeric_columns'])}
📝 Colunas categóricas: {len(dataset_info['categorical_columns'])}
"""
    return info_text

def load_csv_from_path(csv_path):
    global df
    try:
        df = pd.read_csv(csv_path)
        info_text = update_dataset_info(csv_path)
        return True, info_text
    except Exception as e:
        df = None
        return False, f"❌ Erro ao carregar CSV: {str(e)}"

def upload_csv_file(file_path):
    global df
    if file_path is None:
        return False, "❌ Nenhum arquivo foi selecionado"
    try:
        df = pd.read_csv(file_path)
        info_text = update_dataset_info(file_path)
        return True, info_text
    except Exception as e:
        df = None
        return False, f"❌ Erro ao carregar CSV: {str(e)}"

# ----------------------------
# Auxiliar de plots
# ----------------------------

def get_plot_path(analysis_type, column=None):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    col_name = f"_{column}" if column else ""
    filename = f"plots/{analysis_type}{col_name}_{int(pd.Timestamp.now().timestamp())}.png"
    return filename

# ----------------------------
# Tools de análise
# ----------------------------

def get_data_summary():
    global df, dataset_info
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado"})
    summary = {
        "filename": dataset_info.get("filename", "unknown"),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist(),
        "numeric_columns": dataset_info.get("numeric_columns", []),
        "categorical_columns": dataset_info.get("categorical_columns", []),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": {col: f"{(df[col].isnull().sum() / len(df) * 100):.2f}%"
                               for col in df.columns if df[col].isnull().sum() > 0},
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    if dataset_info.get("numeric_columns"):
        summary["numeric_statistics"] = df[dataset_info["numeric_columns"]].describe().to_dict()
    if dataset_info.get("categorical_columns"):
        categorical_info = {}
        for col in dataset_info["categorical_columns"][:5]:
            categorical_info[col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        summary["categorical_info"] = categorical_info
    return json.dumps(summary, default=str)

def analyze_distribution(column: str) -> str:
    global df, dataset_info
    if df is None or column not in df.columns:
        return json.dumps({"status": "error", "message": f"Coluna '{column}' não encontrada."})
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    plot_path = get_plot_path("distribution", column)
    try:
        if is_numeric:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].hist(df[column].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0].set_title(f'Histograma de {column}')
            axes[1].boxplot(df[column].dropna())
            axes[1].set_title(f'Boxplot de {column}')
            df[column].dropna().plot(kind='density', ax=axes[2], color='steelblue')
            axes[2].set_title(f'Densidade de {column}')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            try:
                display(Markdown(f"### Gráfico de Distribuição - Coluna: {column}"))
                display(Image(plot_path))
            except:
                pass
            return json.dumps({
                "status": "success",
                "message": f"Análise de distribuição para {column} concluída.",
                "plot_info": f"Gráfico salvo em {plot_path}."
            }, default=str)
        else:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            plt.title(f'Contagem de Frequência de {column}')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            try:
                display(Markdown(f"### Gráfico de Frequência - Coluna: {column}"))
                display(Image(plot_path))
            except:
                pass
            counts = df[column].value_counts().to_dict()
            return json.dumps({
                "status": "success",
                "message": f"Análise de frequência para {column} concluída. Contagens: {counts}.",
                "plot_info": f"Gráfico salvo em {plot_path}."
            }, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Erro ao plotar distribuição para {column}: {e}"})

def analyze_correlation(columns=None) -> str:
    global df
    if df is None:
        return json.dumps({"status": "error", "message": "Nenhum dataset carregado."})
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return json.dumps({"status": "warning", "message": "Não há colunas numéricas suficientes (mínimo 2) para correlação."})
    try:
        columns = [col for col in (columns if columns else numeric_cols) if col in numeric_cols]
        if len(columns) > 20:
            columns = columns[:20]
        correlation_matrix = df[columns].corr()
        plot_path = get_plot_path("correlation_matrix")
        plt.figure(figsize=(max(10, len(columns)), max(8, len(columns) * 0.8)))
        sns.heatmap(correlation_matrix, annot=len(columns) <= 10, cmap='coolwarm', center=0, fmt=".2f")
        plt.title('Matriz de Correlação entre Variáveis Numéricas')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        try:
            display(Markdown("### Gráfico de Correlação - Heatmap"))
            display(Image(plot_path))
        except:
            pass
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    "var1": correlation_matrix.columns[i],
                    "var2": correlation_matrix.columns[j],
                    "correlation": float(correlation_matrix.iloc[i, j])
                })
        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return json.dumps({
            "status": "success",
            "message": "Matriz de correlação calculada e plotada.",
            "top_10_correlations": corr_pairs[:10],
            "plot_info": f"Gráfico salvo em {plot_path}."
        }, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Erro ao plotar correlação: {e}"})

def detect_outliers(column):
    global df
    if df is None or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return json.dumps({"error": f"Coluna '{column}' não é numérica ou não existe"})
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    plot_path = get_plot_path("outliers", column)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot(df[column].dropna()); axes[0].set_title(f'Boxplot de {column}')
    normal_data = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
    axes[1].scatter(range(len(normal_data)), normal_data[column], alpha=0.5, s=10, label='Normal')
    if len(outliers) > 0:
        axes[1].scatter(outliers.index, outliers[column], color='red', alpha=0.7, s=30, label='Outliers')
    axes[1].set_title('Distribuição com Outliers Destacados'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    try:
        display(Markdown(f"### Gráfico de Detecção de Outliers - Coluna: {column}"))
        display(Image(plot_path))
    except:
        pass
    result = {"total_outliers": int(len(outliers)), "percentage": float((len(outliers) / len(df)) * 100)}
    return json.dumps({
        "status": "success",
        "message": f"Detecção de outliers concluída. {len(outliers)} outliers encontrados.",
        "result": result,
        "plot_info": f"Gráfico salvo em {plot_path}."
    }, default=str)

def compare_groups(column, group_by):
    global df
    if df is None or column not in df.columns or group_by not in df.columns or df[group_by].nunique() > 10:
        return json.dumps({"error": "Erro de coluna ou mais de 10 grupos."})
    plot_path = get_plot_path("group_comparison", f"{column}_vs_{group_by}")
    if pd.api.types.is_numeric_dtype(df[column]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        df.boxplot(column=column, by=group_by, ax=axes[0]); plt.suptitle(""); axes[0].set_title(f'{column} por {group_by}')
        for group in df[group_by].unique()[:5]:
            axes[1].hist(df[df[group_by] == group][column].dropna(), alpha=0.5, label=str(group), bins=30)
        axes[1].set_title(f'Distribuição de {column} por {group_by}'); axes[1].legend()
    else:
        cross_tab = pd.crosstab(df[column], df[group_by])
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0]); axes[0].set_title(f'Crosstab: {column} vs {group_by}')
        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
        cross_tab_pct.plot(kind='bar', stacked=True, ax=axes[1]); axes[1].set_title(f'Distribuição Percentual: {column} por {group_by}')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    try:
        display(Markdown(f"### Gráfico de Comparação de Grupos: {column} por {group_by}"))
        display(Image(plot_path))
    except:
        pass
    return json.dumps({
        "status": "success",
        "message": "Comparação de grupos concluída e plotada.",
        "plot_info": f"Gráfico salvo em {plot_path}."
    }, default=str)

# Placeholders (mantidos do notebook). Se o agente chamar, devem existir.
def analyze_temporal_patterns(time_column, value_column=None):
    return json.dumps({"status": "info", "message": "Função analyze_temporal_patterns não implementada neste módulo."})

def execute_custom_code(code):
    global df
    # CUIDADO: Execução arbitrária; reproduzimos o comportamento do notebook (para compatibilidade).
    # Em produção, isole/valide.
    local_vars = {"df": df, "pd": pd, "np": np}
    try:
        exec(code, {}, local_vars)
        return json.dumps({"status": "success", "message": "Código executado com sucesso."})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Erro na execução do código: {e}"})


# Definição das Tools (Function Calling)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": "Retorna um resumo completo do dataset carregado ...",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_distribution",
            "description": "Analisa e visualiza a distribuição de uma coluna específica...",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string", "description": "Nome da coluna"}},
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlation",
            "description": "Calcula e visualiza a correlação entre variáveis numéricas...",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de colunas (opcional, máximo 20)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_outliers",
            "description": "Detecta outliers em uma coluna numérica usando IQR.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string", "description": "Coluna numérica"}},
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_groups",
            "description": "Compara a distribuição de uma coluna agrupada por outra (máx 10 grupos).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Coluna analisada"},
                    "group_by": {"type": "string", "description": "Coluna de agrupamento"}
                },
                "required": ["column", "group_by"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_temporal_patterns",
            "description": "Analisa padrões temporais quando houver coluna de tempo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_column": {"type": "string"},
                    "value_column": {"type": "string"}
                },
                "required": ["time_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_custom_code",
            "description": "Executa código Python customizado. df disponível.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    }
]

available_functions = {
    "get_data_summary": get_data_summary,
    "analyze_distribution": analyze_distribution,
    "analyze_correlation": analyze_correlation,
    "detect_outliers": detect_outliers,
    "compare_groups": compare_groups,
    "analyze_temporal_patterns": analyze_temporal_patterns,
    "execute_custom_code": execute_custom_code
}
