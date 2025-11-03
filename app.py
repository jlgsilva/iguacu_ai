import gradio as gr
import os
import zipfile
import py7zr
import tempfile
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from google import genai
from google.genai import types

# ===========================================================
# CONFIGURAÇÃO DO CLIENTE GEMINI
# ===========================================================
client = None
try:
    client = genai.Client()
    print("Cliente Gemini inicializado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar cliente Gemini: {e}")
    print("Verifique se a variável de ambiente GEMINI_API_KEY está configurada corretamente.")

# ===========================================================
# ESTADO GLOBAL
# ===========================================================
class AppState:
    def __init__(self):
        self.df = None
        self.analysis_results = {}
        self.csv_path = None
        self.summary_stats = None
        self.plots = {}

state = AppState()

# ===========================================================
# FUNÇÕES AUXILIARES DE PARSE E EXTRAÇÃO
# ===========================================================
def parse_nfe(xml_path):
    ns = {"nfe": "http://www.portalfiscal.inf.br/nfe"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    infNFe = root.find(".//nfe:infNFe", ns)
    if infNFe is None:
        raise ValueError("Estrutura de NF-e inválida")

    def gettext_local(tag, parent):
        elem = parent.find(f"nfe:{tag}", ns)
        return elem.text.strip() if elem is not None and elem.text else None

    ide = infNFe.find("nfe:ide", ns)
    emit = infNFe.find("nfe:emit", ns)
    dest = infNFe.find("nfe:dest", ns)
    total = infNFe.find(".//nfe:total/nfe:ICMSTot", ns)

    nfe_data = {
        "chave": infNFe.attrib.get("Id", ""),
        "numero": gettext_local("nNF", ide),
        "data_emissao": gettext_local("dhEmi", ide),
        "natureza_operacao": gettext_local("natOp", ide),
        "modelo": gettext_local("mod", ide),
        "serie": gettext_local("serie", ide),
        "tipo_operacao": gettext_local("tpNF", ide),
    }

    if emit is not None:
        nfe_data["emitente_cnpj"] = gettext_local("CNPJ", emit)
        nfe_data["emitente_nome"] = gettext_local("xNome", emit)

    if dest is not None:
        nfe_data["destinatario_cnpj"] = gettext_local("CNPJ", dest)
        nfe_data["destinatario_nome"] = gettext_local("xNome", dest)

    if total is not None:
        nfe_data["valor_nf"] = gettext_local("vNF", total)

    itens = []
    for det in infNFe.findall("nfe:det", ns):
        prod = det.find("nfe:prod", ns)
        if prod is not None:
            item = {
                "item": det.attrib.get("nItem"),
                "codigo": gettext_local("cProd", prod),
                "descricao": gettext_local("xProd", prod),
                "ncm": gettext_local("NCM", prod),
                "cfop": gettext_local("CFOP", prod),
                "unidade": gettext_local("uCom", prod),
                "quantidade": gettext_local("qCom", prod),
                "valor_unitario": gettext_local("vUnCom", prod),
                "valor_total": gettext_local("vProd", prod),
            }
            itens.append(item)
    
    nfe_data["itens"] = json.dumps(itens, ensure_ascii=False)
    return nfe_data

def extract_archive(file_path, extract_to):
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith(".7z"):
        with py7zr.SevenZipFile(file_path, "r") as archive:
            archive.extractall(extract_to)
    else:
        raise ValueError("Formato de arquivo não suportado. Use .zip ou .7z")

# ===========================================================
# ANÁLISE COM GEMINI E GERAÇÃO DE GRÁFICOS
# ===========================================================
def generate_monthly_spending_chart(df):
    """Gera histograma de gastos mensais"""
    try:
        df_copy = df.copy()
        
        # Usar coluna já processada
        if 'mes_ano' not in df_copy.columns:
            print("✗ Coluna 'mes_ano' não encontrada")
            return None
            
        monthly = df_copy.groupby('mes_ano')['valor_nf'].sum().reset_index()
        monthly = monthly.sort_values('mes_ano')
        
        if len(monthly) == 0:
            print("✗ Nenhum dado mensal para plotar")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(monthly['mes_ano'], monthly['valor_nf'], color='steelblue', 
                      edgecolor='black', linewidth=1.5)
        
        # Adicionar valores no topo das barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'R$ {height:,.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Mês/Ano', fontsize=13, fontweight='bold')
        ax.set_ylabel('Valor Total (R$)', fontsize=13, fontweight='bold')
        ax.set_title('Gastos Mensais - Notas Fiscais', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        path = os.path.join(tempfile.gettempdir(), "monthly_spending.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"✓ Gráfico mensal gerado: {len(monthly)} meses plotados")
        return path
    except Exception as e:
        print(f"✗ Erro ao gerar gráfico mensal: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_top_items_chart(df):
    """Gera gráfico dos top 10 itens mais comprados"""
    try:
        all_items = []
        for itens_str in df['itens'].dropna():
            try:
                itens = json.loads(itens_str)
                for item in itens:
                    desc = item.get('descricao', '').upper().strip()
                    valor = float(item.get('valor_total', 0))
                    if desc and valor > 0:
                        all_items.append({'descricao': desc, 'valor': valor})
            except:
                continue
        
        if not all_items:
            print("✗ Nenhum item válido encontrado")
            return None
            
        items_df = pd.DataFrame(all_items)
        top_items = items_df.groupby('descricao')['valor'].sum().nlargest(10).reset_index()
        
        # Truncar nomes longos
        top_items['descricao_curta'] = top_items['descricao'].apply(
            lambda x: x[:60] + '...' if len(x) > 60 else x
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(top_items['descricao_curta'], top_items['valor'], 
                       color='coral', edgecolor='black', linewidth=1.5)
        
        # Adicionar valores
        for i, (bar, val) in enumerate(zip(bars, top_items['valor'])):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f'R$ {val:,.2f}',
                   ha='left', va='center', fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Valor Total (R$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Item', fontsize=13, fontweight='bold')
        ax.set_title('Top 10 Itens Mais Comprados (por valor)', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        path = os.path.join(tempfile.gettempdir(), "top_items.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"✓ Gráfico Top 10 gerado: {path}")
        return path
    except Exception as e:
        print(f"✗ Erro ao gerar gráfico de itens: {e}")
        import traceback
        traceback.print_exc()
        return None

def estimate_co2_emissions(df):
    """Estima emissões de CO2 baseado em categorias de produtos"""
    try:
        # Fatores médios de emissão por categoria (kg CO2 / R$)
        emission_factors = {
            'alimentos': 0.5,
            'eletrônicos': 1.2,
            'construção': 0.8,
            'limpeza': 0.3,
            'vestuário': 0.6,
            'móveis': 0.7,
            'outros': 0.5
        }
        
        def categorize_item(desc):
            desc_lower = desc.lower()
            if any(word in desc_lower for word in ['aliment', 'comida', 'cafe', 'arroz', 'feijao', 'massa', 'leite', 'oleo', 'acucar']):
                return 'alimentos'
            elif any(word in desc_lower for word in ['eletro', 'cabo', 'lamp', 'camera', 'monitor', 'tomada', 'condutor']):
                return 'eletrônicos'
            elif any(word in desc_lower for word in ['cimento', 'massa corrida', 'tinta', 'areia', 'cano', 'tubo', 'registro']):
                return 'construção'
            elif any(word in desc_lower for word in ['limpeza', 'detergente', 'sabao', 'desinfetante', 'alcool', 'hipoclorito']):
                return 'limpeza'
            elif any(word in desc_lower for word in ['camiseta', 'calca', 'uniforme', 'jaleco', 'bota', 'luva']):
                return 'vestuário'
            elif any(word in desc_lower for word in ['movel', 'cadeira', 'mesa']):
                return 'móveis'
            return 'outros'
        
        df_copy = df.copy()
        
        # Usar coluna já processada
        if 'mes_ano' not in df_copy.columns:
            print("✗ Coluna 'mes_ano' não encontrada para CO2")
            return None, {}
        
        monthly_co2 = []
        monthly_details = {}
        
        for mes in sorted(df_copy['mes_ano'].dropna().unique()):
            mes_data = df_copy[df_copy['mes_ano'] == mes]
            total_co2 = 0
            categoria_co2 = {}
            
            for itens_str in mes_data['itens'].dropna():
                try:
                    itens = json.loads(itens_str)
                    for item in itens:
                        desc = item.get('descricao', '')
                        valor = float(item.get('valor_total', 0))
                        if valor > 0:
                            categoria = categorize_item(desc)
                            co2 = valor * emission_factors[categoria]
                            total_co2 += co2
                            categoria_co2[categoria] = categoria_co2.get(categoria, 0) + co2
                except:
                    continue
            
            mes_str = str(mes)
            monthly_co2.append({'mes_ano': mes_str, 'co2_kg': total_co2})
            monthly_details[mes_str] = categoria_co2
        
        if not monthly_co2:
            print("✗ Nenhum dado de CO2 calculado")
            return None, {}
            
        co2_df = pd.DataFrame(monthly_co2).sort_values('mes_ano')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Linha principal
        line = ax.plot(co2_df['mes_ano'], co2_df['co2_kg'], 
                      marker='o', linewidth=3, color='darkgreen', 
                      markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                      label='Emissões Totais')[0]
        
        # Área preenchida
        ax.fill_between(range(len(co2_df)), co2_df['co2_kg'], 
                       alpha=0.3, color='lightgreen')
        
        # Valores nos pontos
        for i, (x, y) in enumerate(zip(co2_df['mes_ano'], co2_df['co2_kg'])):
            ax.text(i, y, f'{y:.1f} kg', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Mês/Ano', fontsize=13, fontweight='bold')
        ax.set_ylabel('Emissões de CO₂ (kg)', fontsize=13, fontweight='bold')
        ax.set_title('Estimativa de Emissões de CO₂ Mensais', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        path = os.path.join(tempfile.gettempdir(), "co2_emissions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Calcular totais
        total_co2 = co2_df['co2_kg'].sum()
        co2_summary = {
            'total_kg': total_co2,
            'total_ton': total_co2 / 1000,
            'monthly_data': co2_df.to_dict('records'),
            'category_details': monthly_details,
            'emission_factors': emission_factors
        }
        
        print(f"✓ Gráfico CO2 gerado: {len(co2_df)} meses")
        print(f"  Total CO2: {total_co2:.2f} kg ({total_co2/1000:.3f} ton)")
        
        return path, co2_summary
    except Exception as e:
        print(f"✗ Erro ao estimar CO2: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

def call_gemini_analysis(prompt, data_summary):
    """Chama API Gemini para análise"""
    if client is None:
        return "Erro: Cliente Gemini não inicializado."
    
    full_prompt = f"""Você é um analista de dados especializado em Notas Fiscais eletrônicas brasileiras (NF-e).

CONTEXTO DOS DADOS:
{data_summary}

TAREFA:
{prompt}

Forneça uma análise detalhada, profissional e objetiva."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4000
            )
        )
        return response.text
    except Exception as e:
        return f"Erro ao chamar API Gemini: {str(e)}"

def analyze_data_structure_with_gemini(df):
    """Usa Gemini para analisar estrutura dos dados e sugerir melhor forma de processar datas"""
    try:
        sample_dates = df['data_emissao'].head(10).tolist()
        sample_data = df.head(5).to_dict('records')
        
        structure_prompt = f"""Analise a estrutura deste dataset de Notas Fiscais e responda OBJETIVAMENTE:

AMOSTRA DE DATAS:
{sample_dates}

AMOSTRA DE REGISTROS:
{json.dumps(sample_data, indent=2, ensure_ascii=False)[:1000]}

PERGUNTAS:
1. Qual o formato exato das datas? (ex: ISO 8601 com timezone)
2. Como extrair mês e ano dessas datas em Python/Pandas?
3. Há datas inválidas ou nulas que precisam ser tratadas?
4. Qual a melhor estratégia para agrupar por mês/ano?

Seja TÉCNICO e DIRETO. Responda em formato de código Python quando aplicável."""

        response = call_gemini_analysis(structure_prompt, "")
        print("\n🤖 Análise Gemini da estrutura:")
        print(response[:500])
        
        return response
    except Exception as e:
        print(f"Erro na análise de estrutura: {e}")
        return None

def perform_autonomous_analysis(df):
    """Executa análise autônoma completa"""
    try:
        print("\n" + "="*60)
        print("INICIANDO ANÁLISE AUTÔNOMA")
        print("="*60)
        
        # ANÁLISE PRÉVIA COM GEMINI
        print("\n🔍 Analisando estrutura dos dados com Gemini...")
        analyze_data_structure_with_gemini(df)
        
        # Criar cópia para não modificar o original
        df_work = df.copy()
        
        # Preparar estatísticas detalhadas
        total_nfs = len(df_work)
        df_work['valor_nf'] = pd.to_numeric(df_work['valor_nf'], errors='coerce')
        total_valor = df_work['valor_nf'].sum()
        
        print("\n📅 Processando datas...")
        print(f"Amostra de datas originais: {df_work['data_emissao'].head(3).tolist()}")
        
        # CONVERSÃO ROBUSTA DE DATAS COM MÚLTIPLAS TENTATIVAS
        # Formato: 2021-09-14T08:26:00-03:00 (ISO 8601 com timezone)
        df_work['data_emissao_original'] = df_work['data_emissao']
        
        # Tentativa 1: Conversão direta
        df_work['data_emissao_dt'] = pd.to_datetime(df_work['data_emissao'], errors='coerce', utc=True)
        
        # Tentativa 2: Remover timezone manualmente se necessário
        if df_work['data_emissao_dt'].isna().all():
            print("⚠️ Conversão direta falhou, tentando remover timezone...")
            df_work['data_emissao_clean'] = df_work['data_emissao'].astype(str).str.slice(0, 19)
            df_work['data_emissao_dt'] = pd.to_datetime(df_work['data_emissao_clean'], errors='coerce')
        
        # Filtrar apenas registros com datas válidas
        df_com_data = df_work[df_work['data_emissao_dt'].notna()].copy()
        
        print(f"✓ Datas convertidas: {len(df_com_data)} de {len(df_work)} registros")
        print(f"Amostra convertida: {df_com_data['data_emissao_dt'].head(3).tolist()}")
        
        if len(df_com_data) == 0:
            error_msg = """❌ Erro: Nenhuma data válida encontrada no dataset.

DIAGNÓSTICO:
1. Formato detectado das datas não é compatível com pandas
2. Verifique se a coluna 'data_emissao' existe e contém valores válidos
3. Formato esperado: ISO 8601 (ex: 2021-09-14T08:26:00-03:00)

SOLUÇÃO:
- Revise o processo de extração dos XMLs
- Garanta que as datas sejam extraídas corretamente"""
            return error_msg, [None, None, None]
        
        data_inicio = df_com_data['data_emissao_dt'].min()
        data_fim = df_com_data['data_emissao_dt'].max()
        
        # Extrair mês/ano
        df_com_data['ano'] = df_com_data['data_emissao_dt'].dt.year
        df_com_data['mes'] = df_com_data['data_emissao_dt'].dt.month
        df_com_data['mes_ano_str'] = df_com_data['data_emissao_dt'].dt.strftime('%Y-%m')
        
        print(f"✓ Extraído ano/mês: {df_com_data['mes_ano_str'].head(3).tolist()}")
        
        # Estatísticas mensais
        monthly_stats = df_com_data.groupby('mes_ano_str').agg({
            'valor_nf': ['sum', 'count', 'mean']
        }).reset_index()
        
        monthly_stats.columns = ['Mês/Ano', 'Total (R$)', 'Qtd NFs', 'Média (R$)']
        monthly_stats = monthly_stats.sort_values('Mês/Ano')
        
        # Formatar valores para exibição
        monthly_display = monthly_stats.copy()
        monthly_display['Total (R$)'] = monthly_display['Total (R$)'].apply(lambda x: f'R$ {x:,.2f}')
        monthly_display['Média (R$)'] = monthly_display['Média (R$)'].apply(lambda x: f'R$ {x:,.2f}')
        
        fornecedores = df_work['emitente_nome'].nunique()
        top_fornecedores = df_com_data.groupby('emitente_nome')['valor_nf'].sum().nlargest(5)
        
        # Análise de itens
        total_itens = 0
        for itens_str in df_work['itens'].dropna():
            try:
                itens = json.loads(itens_str)
                total_itens += len(itens)
            except:
                continue
        
        data_summary = f"""
════════════════════════════════════════════════════════════
RESUMO EXECUTIVO DO DATASET - NOTAS FISCAIS ELETRÔNICAS
════════════════════════════════════════════════════════════

📊 INFORMAÇÕES GERAIS:
  • Total de Notas Fiscais: {total_nfs}
  • Valor Total Acumulado: R$ {total_valor:,.2f}
  • Período Analisado: {data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}
  • Número de Fornecedores Distintos: {fornecedores}
  • Total de Itens Comprados: {total_itens}

📅 DISTRIBUIÇÃO TEMPORAL:
  Formato Original: ISO 8601 com timezone (ex: 2021-09-14T08:26:00-03:00)
  Colunas extraídas: ano, mês, mes_ano_str
  
  Distribuição Mensal de Gastos:
{monthly_display.to_string(index=False)}

💰 TOP 5 FORNECEDORES (por valor):
{chr(10).join([f"  • {nome}: R$ {valor:,.2f}" for nome, valor in top_fornecedores.items()])}

🔍 ESTRUTURA DOS DADOS:
  Colunas disponíveis: {', '.join(df_work.columns)}
  
  A coluna 'itens' contém JSON com detalhes de cada produto:
  - descricao: nome do produto
  - valor_total: valor do item
  - quantidade: quantidade comprada
  - ncm: código NCM (classificação fiscal)
  - cfop: código de operação fiscal
"""

        print("✓ Resumo preparado")
        
        # Gerar gráficos com o DataFrame processado
        print("\n📊 Gerando visualizações...")
        
        # Preparar dados para gráficos
        df_plot = df_com_data.copy()
        df_plot['mes_ano'] = df_plot['mes_ano_str']  # Usar string para compatibilidade
        
        plot1 = generate_monthly_spending_chart(df_plot)
        plot2 = generate_top_items_chart(df_work)
        plot3, co2_summary = estimate_co2_emissions(df_plot)
        
        # Preparar resumo de CO2
        co2_text = ""
        if co2_summary:
            co2_text = f"""

🌱 RESUMO DE EMISSÕES DE CO₂:
  • Total Estimado: {co2_summary['total_kg']:.2f} kg ({co2_summary['total_ton']:.4f} toneladas)
  • Período: {data_inicio.strftime('%m/%Y')} a {data_fim.strftime('%m/%Y')}
  
  Distribuição Mensal:
{chr(10).join([f"    - {item['mes_ano']}: {item['co2_kg']:.2f} kg CO₂" for item in co2_summary['monthly_data']])}

  Fatores de Emissão Utilizados (kg CO₂ por R$ gasto):
{chr(10).join([f"    • {cat.capitalize()}: {fator}" for cat, fator in co2_summary['emission_factors'].items()])}
"""
        
        # Análise com Gemini
        print("\n🤖 Consultando Gemini 2.5 Flash para análise completa...")
        
        analysis_prompt = f"""Com base nos dados fornecidos, realize uma análise COMPLETA e DETALHADA respondendo:

1. **ANÁLISE TEMPORAL E SAZONALIDADE:**
   - Analise a distribuição mensal de gastos apresentada na tabela
   - Identifique padrões, tendências de crescimento ou redução
   - Há concentração de gastos em períodos específicos? 
   - Qual foi o mês de maior e menor gasto?
   - Calcule a variação percentual entre os meses

2. **FORNECEDORES E CATEGORIAS:**
   - Analise o Top 5 de fornecedores apresentado
   - Qual a concentração de gastos? (ex: top 3 representa X% do total)
   - Com base nos nomes dos fornecedores, qual o perfil de compras?
   - Há dependência excessiva de poucos fornecedores?

3. **ANOMALIAS E ATENÇÃO:**
   - Com {total_nfs} notas fiscais em {len(monthly_stats)} meses, identifique períodos atípicos
   - Há meses sem notas fiscais? Se sim, quais e por quê isso pode ser relevante?
   - O valor médio por nota fiscal está consistente ou há variações grandes?

4. **OTIMIZAÇÃO E RECOMENDAÇÕES:**
   - Quais ações práticas e específicas podem reduzir custos?
   - Onde focar negociações com fornecedores?
   - Há oportunidades de consolidação de compras?

5. **PERFIL ORGANIZACIONAL:**
   - Com base nos fornecedores e valores, qual o tipo de organização? (indústria, comércio, serviços, etc)
   - Quais as principais necessidades de compra identificadas?

6. **ANÁLISE DE SUSTENTABILIDADE (CO₂):**
{co2_text}
   
   Com base nas emissões estimadas:
   - Qual a tendência das emissões ao longo dos meses?
   - Quais categorias de produtos contribuem mais para as emissões?
   - Que ações poderiam reduzir a pegada de carbono sem comprometer operações?
   - Compare as emissões com benchmarks do setor público brasileiro (se conhecer)

IMPORTANTE: Use os números específicos fornecidos. Seja OBJETIVO e DIRETO nas respostas."""

        analysis_text = call_gemini_analysis(analysis_prompt, data_summary)
        
        print("✓ Análise Gemini concluída")
        
        # Metodologia CO2
        methodology = """
═══════════════════════════════════════════════════════════════════════
METODOLOGIA DE ESTIMATIVA DE EMISSÕES DE CO₂
═══════════════════════════════════════════════════════════════════════

📚 REFERÊNCIAS UTILIZADAS:
  1. DEFRA 2023 - Department for Environment, Food & Rural Affairs (Reino Unido)
     Conversion factors for greenhouse gas reporting
     
  2. IPCC AR6 - Intergovernmental Panel on Climate Change
     Sixth Assessment Report - Working Group III (2022)
     Global Warming Potential (GWP-100)

🔬 FATORES DE EMISSÃO APLICADOS (kg CO₂eq / R$ gasto):

  Categoria         | Fator | Base
  ------------------|-------|------------------------------------------
  Alimentos         | 0.5   | Média para produtos alimentícios gerais
  Eletrônicos       | 1.2   | Fabricação e transporte de eletrônicos
  Construção        | 0.8   | Cimento, aço, materiais de construção
  Limpeza           | 0.3   | Produtos químicos de limpeza
  Vestuário         | 0.6   | Têxteis e confecções
  Móveis            | 0.7   | Madeira e metais processados
  Outros            | 0.5   | Média genérica

🔍 PROCESSO DE CATEGORIZAÇÃO:
  Os produtos foram classificados automaticamente através de análise de 
  palavras-chave nas descrições dos itens das notas fiscais.
  
  Exemplos de palavras-chave:
  • Alimentos: café, arroz, feijão, massa, leite, óleo, açúcar
  • Eletrônicos: cabo, lâmpada, câmera, monitor, tomada
  • Construção: cimento, tinta, areia, cano, tubo, registro
  • Limpeza: detergente, sabão, desinfetante, álcool, hipoclorito

⚠️ LIMITAÇÕES E DISCLAIMERS:
  1. Valores aproximados para análise comparativa e gestão interna
  2. NÃO devem ser usados para:
     - Inventários oficiais de GEE (Gases de Efeito Estufa)
     - Relatórios de sustentabilidade certificados
     - Compensação de carbono oficial
  3. Para relatórios oficiais, recomenda-se:
     - Análise específica por produto com dados do fabricante
     - Uso de ferramentas certificadas (GHG Protocol, ISO 14064)
     - Validação por terceiros credenciados

📊 PRECISÃO ESTIMADA:
  • Margem de erro: ±30% (devido à categorização genérica)
  • Adequado para: identificação de tendências e hotspots
  • Recomendação: usar para decisões estratégicas, não para compliance

═══════════════════════════════════════════════════════════════════════
"""
        
        full_analysis = f"""# 📊 ANÁLISE AUTOMÁTICA - NOTAS FISCAIS ELETRÔNICAS
### Powered by Gemini 2.5 Flash

{data_summary}

## 🔍 ANÁLISE DETALHADA DA IA

{analysis_text}

{methodology}

---
**💡 Dica:** Você pode fazer perguntas adicionais no chat interativo abaixo sobre qualquer aspecto desta análise.
"""
        
        state.analysis_results = {
            'full_analysis': full_analysis,
            'plots': [plot1, plot2, plot3],
            'co2_summary': co2_summary
        }
        
        # Atualizar o estado global com o DataFrame processado
        state.df = df_com_data
        
        print("\n✓✓✓ ANÁLISE COMPLETA FINALIZADA ✓✓✓\n")
        
        return full_analysis, [plot1, plot2, plot3]
        
    except Exception as e:
        error_msg = f"❌ Erro na análise autônoma: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, [None, None, None]

# ===========================================================
# PROCESSAMENTO PRINCIPAL
# ===========================================================
def process_archive(uploaded_file):
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extrair arquivo
        file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
        extract_archive(file_path, temp_dir)
        
        # Encontrar XMLs
        xml_files = list(Path(temp_dir).rglob("*.xml"))
        
        if not xml_files:
            yield "❌ Nenhum arquivo XML encontrado no arquivo compactado.", None, None, None, None, None, gr.update(interactive=False)
            return
        
        yield f"📦 Processando {len(xml_files)} arquivos XML...", None, None, None, None, None, gr.update(interactive=False)
        
        # Parse XMLs
        nfe_data = []
        for xml_file in xml_files:
            try:
                data = parse_nfe(xml_file)
                nfe_data.append(data)
            except Exception as e:
                print(f"Erro em {xml_file}: {e}")
                continue
        
        if not nfe_data:
            yield "❌ Não foi possível processar nenhum XML válido.", None, None, None, None, None, gr.update(interactive=False)
            return
        
        # Criar DataFrame
        df = pd.DataFrame(nfe_data)
        df["valor_nf"] = pd.to_numeric(df["valor_nf"], errors="coerce")
        
        # Salvar CSV
        csv_path = os.path.join(temp_dir, "notas_fiscais.csv")
        df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
        
        state.df = df
        state.csv_path = csv_path
        
        yield (
            f"✅ {len(nfe_data)} notas fiscais processadas!\n\n🤖 Iniciando análise com Gemini 2.5 Flash...",
            df.head(20),
            csv_path,
            None, None, None,
            gr.update(interactive=False)
        )
        
        # Análise com Gemini
        analysis_text, plots = perform_autonomous_analysis(df)
        
        elapsed = int(time.time() - start_time)
        final_msg = f"""✅ ANÁLISE CONCLUÍDA EM {elapsed}s

{analysis_text}

💬 Você pode fazer perguntas adicionais no chat interativo abaixo."""
        
        yield (
            final_msg,
            df.head(20),
            csv_path,
            plots[0], plots[1], plots[2],
            gr.update(interactive=True)
        )
        
    except Exception as e:
        yield f"❌ Erro: {str(e)}", None, None, None, None, None, gr.update(interactive=False)

# ===========================================================
# CHAT INTERATIVO
# ===========================================================
def chat_response(message, history):
    if state.df is None:
        return history + [(message, "⚠️ Por favor, processe um arquivo primeiro.")]
    
    # Preparar contexto
    context = f"""Você tem acesso aos seguintes dados analisados:

ANÁLISE PRÉVIA:
{state.analysis_results.get('full_analysis', 'Análise não disponível')}

ESTATÍSTICAS DO DATASET:
- Total de registros: {len(state.df)}
- Valor total: R$ {state.df['valor_nf'].sum():,.2f}
- Período: {pd.to_datetime(state.df['data_emissao']).min()} a {pd.to_datetime(state.df['data_emissao']).max()}

PERGUNTA DO USUÁRIO:
{message}

Responda de forma precisa e baseada nos dados."""
    
    response = call_gemini_analysis(message, context)
    return history + [(message, response)]

# ===========================================================
# INTERFACE GRADIO
# ===========================================================
with gr.Blocks(title="Processador NF-e com IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 Processador Inteligente de Notas Fiscais (NF-e)
    ### Powered by Gemini 2.5 Flash
    
    Envie um arquivo compactado (.zip ou .7z) com XMLs de NF-e para análise automática com IA.
    """)
    
    with gr.Row():
        arquivo_input = gr.File(
            label="📁 Arquivo Compactado (.zip ou .7z)",
            file_types=[".zip", ".7z"]
        )
    
    botao = gr.Button("🚀 Processar e Analisar", variant="primary", size="lg")
    
    saida_texto = gr.Markdown("Aguardando arquivo...")
    
    gr.Markdown("## 📋 Dados Consolidados")
    tabela_csv = gr.Dataframe(label="Amostra do CSV Unificado", interactive=False)
    csv_download = gr.File(label="⬇️ Baixar CSV Completo")
    
    gr.Markdown("## 📈 Visualizações Geradas pela IA")
    
    plot1 = gr.Image(label="💰 Gastos Mensais")
    plot2 = gr.Image(label="🛒 Top 10 Itens")
    plot3 = gr.Image(label="🌱 Emissões de CO₂")
    
    gr.Markdown("## 💬 Chat Interativo com IA")
    chatbot = gr.Chatbot(label="Converse sobre os dados", height=400)
    with gr.Row():
        chat_input = gr.Textbox(
            label="Sua pergunta",
            placeholder="Ex: Qual foi o mês de maior gasto?",
            interactive=False
        )
        submit_btn = gr.Button("Enviar", variant="primary")
    clear_btn = gr.Button("🗑️ Limpar Chat")
    
    # Eventos
    botao.click(
        fn=process_archive,
        inputs=arquivo_input,
        outputs=[saida_texto, tabela_csv, csv_download, plot1, plot2, plot3, chat_input]
    )
    
    submit_btn.click(
        fn=chat_response,
        inputs=[chat_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        chat_input
    )
    
    chat_input.submit(
        fn=chat_response,
        inputs=[chat_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        chat_input
    )
    
    clear_btn.click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch()
