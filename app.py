# app.py
import os
import gradio as gr
import pandas as pd

from tools import (
    df, dataset_info, upload_csv_file, load_csv_from_path
)
from agent import (
    memory, run_agent_core, autonomous_analysis
)

# Helpers de status e interações

def get_dataset_status():
    from tools import df, dataset_info
    if df is None:
        return "❌ **Nenhum dataset carregado**", None
    status = f"""✅ **Dataset Carregado**

📁 **Arquivo:** {dataset_info.get('filename', 'N/A')}
📊 **Dimensões:** {df.shape[0]:,} linhas × {df.shape[1]} colunas
💾 **Memória:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
🔢 **Colunas Numéricas:** {len(dataset_info.get('numeric_columns', []))}
📝 **Colunas Categóricas:** {len(dataset_info.get('categorical_columns', []))}
"""
    preview = df.head(5)
    return status, preview

def process_query(message, history):
    from tools import df
    if df is None:
        history = history or []
        history.append([message, "⚠️ Por favor, carregue um dataset primeiro."])
        return history
    try:
        updated_history = run_agent_core(message, history=history)
        return updated_history
    except Exception as e:
        history.append([message, f"❌ **Erro:** {str(e)}\n\nTente reformular sua pergunta ou resete a memória."])
        return history

def upload_and_update(file_path):
    if not file_path:
        return (
            "❌ Nenhum arquivo selecionado",
            None,
            "❌ **Nenhum dataset carregado**",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            "Nenhuma análise autônoma foi realizada."
        )
    success, message = upload_csv_file(file_path)
    if success:
        status, preview = get_dataset_status()
        autonomy_history, final_conclusion = autonomous_analysis()
        return (
            f"✅ **Sucesso!**\n\n{message}\n\n⚠️ **Análise Autônoma Iniciada (veja os gráficos gerados em plots/)!**",
            preview,
            status,
            gr.update(interactive=True),
            gr.update(interactive=True),
            autonomy_history,
            final_conclusion
        )
    else:
        return (
            message, None, "❌ **Erro ao carregar dataset**",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            "Nenhuma análise autônoma foi realizada."
        )

def load_kaggle_and_update():
    # Opcional: habilitar quando configurar kagglehub e dataset
    try:
        import kagglehub
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = os.path.join(path, "creditcard.csv")
        success, message = load_csv_from_path(csv_path)
        if success:
            status, preview = get_dataset_status()
            autonomy_history, final_conclusion = autonomous_analysis()
            return (
                f"✅ **Dataset do Kaggle carregado!**\n\n{message}\n\n⚠️ **Análise Autônoma Iniciada (veja os gráficos gerados em plots/)!**",
                preview,
                status,
                gr.update(interactive=True),
                gr.update(interactive=True),
                autonomy_history,
                final_conclusion
            )
        else:
            return (
                f"❌ {message}", None, "❌ **Erro ao carregar dataset**",
                gr.update(interactive=False),
                gr.update(interactive=False),
                [],
                "Nenhuma análise autônoma foi realizada."
            )
    except Exception as e:
        error_msg = f"❌ Erro ao carregar dataset do Kaggle: {str(e)}"
        return (
            error_msg, None, "❌ **Erro ao carregar dataset**",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            "Nenhuma análise autônoma foi realizada."
        )

def clear_chat_history():
    return []

def reset_and_notify():
    memory.reset()
    return [], "✅ Memória resetada! O dataset e as conclusões anteriores foram limpos. O chat está pronto para novas análises."

def get_detailed_info():
    from tools import df, dataset_info
    if df is None:
        return "⚠️ Nenhum dataset carregado"
    info = f"""## 📊 Informações Detalhadas do Dataset

### 📋 Informações Gerais
- **Arquivo:** {dataset_info.get('filename', 'N/A')}
- **Dimensões:** {df.shape[0]:,} linhas × {df.shape[1]} colunas
- **Memória:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Duplicatas:** {df.duplicated().sum():,}

### 📊 Resumo das Colunas

"""
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        emoji = "🔢" if pd.api.types.is_numeric_dtype(df[col]) else "📝"
        info += f"#### {emoji} **{col}**\n"
        info += f"- **Tipo:** `{dtype}`\n"
        info += f"- **Valores únicos:** {unique:,}\n"
        info += f"- **Valores faltantes:** {missing:,} ({missing_pct:.2f}%)\n"
        if pd.api.types.is_numeric_dtype(df[col]):
            info += f"- **Min:** {df[col].min():.2f} | **Max:** {df[col].max():.2f} | **Média:** {df[col].mean():.2f}\n"
        else:
            top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            info += f"- **Valor mais frequente:** {top_value}\n"
        info += "\n"
    return info

def update_history():
    if not memory.analysis_results:
        return "⚠️ Nenhuma análise realizada ainda"
    history_text = "## 📜 Histórico de Análises Realizadas\n\n"
    for i, (analysis_type, params) in enumerate(memory.analysis_results.items(), 1):
        history_text += f"**{i}. {analysis_type}**\n"
        if params:
            param_str = str(params)
            if len(param_str) > 150:
                param_str = param_str[:150] + "..."
            history_text += f"  - Parâmetros: `{param_str}`\n"
        history_text += "\n"
    return history_text

def update_conclusions():
    if not memory.conclusions:
        return """⚠️ Nenhuma conclusão registrada ainda.

💡 Dica: Pergunte ao agente:
- "Quais suas conclusões sobre os dados?"
- "Faça um resumo dos principais achados"
- "Quais insights você obteve das análises?"
"""
    conclusions_text = "## 🎯 Conclusões e Insights do Agente\n\n"
    for i, conclusion in enumerate(memory.conclusions, 1):
        conclusions_text += f"### 📌 Conclusão {i}\n{conclusion}\n\n---\n\n"
    return conclusions_text

# Interface Gradio
with gr.Blocks(
    title="🤖 Agente EDA Autônomo",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    css="""
        .gradio-container {font-family: 'Segoe UI', sans-serif; max-width: 1400px !important;}
        .header {
            text-align: center; padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 15px; margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .section-header {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            color: white; padding: 12px 20px; border-radius: 8px;
            margin: 15px 0 10px 0; font-weight: bold;
        }
        .status-box { padding: 15px; border-radius: 10px; border: 2px solid #e0e0e0; background: #f8f9fa; margin: 10px 0; }
        .suggestion-btn { margin: 5px 0 !important; text-align: left !important; }
    """
) as demo:

    gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5em;">🤖 Agente de Análise Exploratória de Dados Autônomo</h1>
            <p style="font-size: 1.2em; margin-top: 15px; opacity: 0.95;">
                Análise inteligente e automatizada de qualquer dataset CSV usando GPT-4o
            </p>
        </div>
    """)

    gr.HTML('<div class="section-header">📥 PASSO 1: Carregue seu Dataset e Dispare a Análise Autônoma</div>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📤 Opção A: Upload de Arquivo
            Faça upload de qualquer arquivo CSV do seu computador.
            """)
            file_upload = gr.File(label="Selecione o arquivo CSV", file_types=[".csv"], type="filepath")
            upload_btn = gr.Button("📥 Carregar Arquivo CSV", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📦 Opção B: Dataset de Exemplo
            Use o dataset de fraude em cartão de crédito do Kaggle para testar.
            """)
            gr.Markdown("**Dataset:** Credit Card Fraud Detection \n**Tamanho:** ~150MB, 284,807 transações")
            kaggle_btn = gr.Button("📦 Carregar Dataset do Kaggle", variant="secondary", size="lg")

    with gr.Row():
        with gr.Column():
            upload_status = gr.Markdown(value="⚠️ **Aguardando upload de dataset...**")
        with gr.Column():
            dataset_status_box = gr.Markdown(value="❌ **Nenhum dataset carregado**")

    dataset_preview = gr.Dataframe(
        label="📊 Preview do Dataset (primeiras 5 linhas)",
        wrap=True,
        interactive=False
    )

    gr.HTML('<div class="section-header" style="background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);">🎯 CONCLUSÃO DA ANÁLISE AUTÔNOMA (Fase 1)</div>')

    autonomous_conclusion_display = gr.Markdown(
        value="A conclusão será exibida aqui após o carregamento e análise do dataset. 🧠",
        label="Resumo e Insights do Agente"
    )

    gr.HTML('<div class="section-header">💬 PASSO 2: Faça Perguntas Adicionais (o Agente Lembra do Contexto!)</div>')

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Conversação com o Agente",
                height=500,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False,
                show_copy_button=True
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Sua pergunta (digite livremente ou use as sugestões ao lado)",
                    placeholder="Ex: Qual foi a correlação mais forte que você encontrou?",
                    lines=3,
                    scale=5,
                    interactive=False
                )
                submit_btn = gr.Button("Enviar 📤", variant="primary", scale=1, size="lg", interactive=False)
            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Limpar Chat", size="sm")
                clear_memory_btn = gr.Button("🧹 Resetar Memória", size="sm")
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Perguntas Sugeridas")
            gr.Markdown("*Clique para preencher o campo:*")
            suggestion_1 = gr.Button("📋 Repita o resumo da análise", size="sm", elem_classes="suggestion-btn")
            suggestion_2 = gr.Button("🔍 Me mostre os outliers da coluna 'Amount'", size="sm", elem_classes="suggestion-btn")
            suggestion_3 = gr.Button("📊 Distribuição das variáveis V1 e V2", size="sm", elem_classes="suggestion-btn")
            suggestion_4 = gr.Button("🔗 Quais as 5 maiores correlações positivas?", size="sm", elem_classes="suggestion-btn")
            suggestion_5 = gr.Button("✍️ Execute um código: print(df.shape)", size="sm", elem_classes="suggestion-btn")
            suggestion_6 = gr.Button("👥 Compare 'Time' pela coluna 'Class'", size="sm", elem_classes="suggestion-btn")
            suggestion_7 = gr.Button("📈 Quais padrões e tendências você identificou?", size="sm", elem_classes="suggestion-btn")
            suggestion_8 = gr.Button("🎯 Quais são suas conclusões finais sobre este dataset?", size="sm", elem_classes="suggestion-btn")
            gr.Markdown("---")
            gr.Markdown("### 📈 Dicas de Uso")
            gr.Markdown("""
            - 🧠 O Agente lembra de todas as análises autônomas.
            - ❓ Faça perguntas específicas (ex: 'coluna X')
            - 🛠️ Use a sugestão de código customizado para análises manuais.
            """)

    with gr.Accordion("📊 Informações Detalhadas do Dataset", open=False):
        refresh_info_btn = gr.Button("🔄 Atualizar Informações")
        detailed_info = gr.Markdown(value="⚠️ Carregue um dataset primeiro")
        refresh_info_btn.click(get_detailed_info, outputs=detailed_info)

    with gr.Accordion("📜 Histórico de Análises (Memória)", open=False):
        refresh_history_btn = gr.Button("🔄 Atualizar Histórico")
        history_display = gr.Markdown(value="⚠️ Nenhuma análise registrada ainda")
        refresh_history_btn.click(update_history, outputs=history_display)

    with gr.Accordion("🎯 Conclusões e Insights Registrados (Memória)", open=False):
        refresh_conclusions_btn = gr.Button("🔄 Atualizar Conclusões")
        conclusions_display = gr.Markdown(value="⚠️ Nenhuma conclusão registrada ainda")
        refresh_conclusions_btn.click(update_conclusions, outputs=conclusions_display)

    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🤖 Agente EDA Autônomo</strong> | Powered by OpenAI GPT-4o & Gradio</p>
        <p style="font-size: 0.9em; margin-top: 10px;">
            💡 <strong>Fluxo de uso:</strong> 1️⃣ Carregue um CSV para iniciar a <strong>Análise Autônoma</strong> → 2️⃣ Revise a Conclusão → 3️⃣ Faça perguntas adicionais no Chat!
        </p>
    </div>
    """)

    # Handlers
    upload_btn.click(
        fn=upload_and_update,
        inputs=[file_upload],
        outputs=[upload_status, dataset_preview, dataset_status_box, msg, submit_btn, chatbot, autonomous_conclusion_display]
    )

    kaggle_btn.click(
        fn=load_kaggle_and_update,
        outputs=[upload_status, dataset_preview, dataset_status_box, msg, submit_btn, chatbot, autonomous_conclusion_display]
    )

    def user_message(message, history):
        if not message or not message.strip():
            return message, history
        return "", history + [[message, None]]

    def bot_response(history):
        if not history or history[-1][1] is not None:
            return history
        user_msg = history[-1][0]
        updated_history = process_query(user_msg, history[:-1])
        return updated_history

    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )

    clear_chat_btn.click(clear_chat_history, outputs=chatbot)
    clear_memory_btn.click(reset_and_notify, outputs=[chatbot, autonomous_conclusion_display])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 INICIANDO INTERFACE GRADIO...")
    print("="*80)
    demo.launch(
        share=False,
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
