import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def recencia_class(x, r, q_dict):
    if x <= q_dict[r][0.25]:
        return 'A'
    elif x <= q_dict[r][0.50]:
        return 'B'
    elif x <= q_dict[r][0.75]:
        return 'C'
    else:
        return 'D'

def freq_val_class(x, fv, q_dict):
    if x <= q_dict[fv][0.25]:
        return 'D'
    elif x <= q_dict[fv][0.50]:
        return 'C'
    elif x <= q_dict[fv][0.75]:
        return 'B'
    else:
        return 'A'

def encontrar_k_otimo(df_scaled, max_k=10):
    """
    Encontra o número ótimo de clusters usando o método do cotovelo e silhouette score
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
    
    return k_range, inertias, silhouette_scores

def interpretar_clusters(df_rfv_clusters):
    """
    Interpreta os clusters baseado nas médias de R, F, V
    """
    interpretacoes = {}
    
    for cluster in df_rfv_clusters['Cluster'].unique():
        cluster_data = df_rfv_clusters[df_rfv_clusters['Cluster'] == cluster]
        
        media_recencia = cluster_data['Recencia'].mean()
        media_frequencia = cluster_data['Frequencia'].mean()
        media_valor = cluster_data['Valor'].mean()
        
        # Classificação baseada nas médias
        if media_recencia <= df_rfv_clusters['Recencia'].quantile(0.33):
            recencia_cat = "Baixa"
        elif media_recencia <= df_rfv_clusters['Recencia'].quantile(0.66):
            recencia_cat = "Média"
        else:
            recencia_cat = "Alta"
            
        if media_frequencia <= df_rfv_clusters['Frequencia'].quantile(0.33):
            frequencia_cat = "Baixa"
        elif media_frequencia <= df_rfv_clusters['Frequencia'].quantile(0.66):
            frequencia_cat = "Média"
        else:
            frequencia_cat = "Alta"
            
        if media_valor <= df_rfv_clusters['Valor'].quantile(0.33):
            valor_cat = "Baixo"
        elif media_valor <= df_rfv_clusters['Valor'].quantile(0.66):
            valor_cat = "Médio"
        else:
            valor_cat = "Alto"
        
        # Definir perfil do cluster
        if recencia_cat == "Baixa" and frequencia_cat == "Alta" and valor_cat == "Alto":
            perfil = "🌟 Clientes VIP"
            acao = "Programa de fidelidade premium, produtos exclusivos"
        elif recencia_cat == "Baixa" and (frequencia_cat == "Alta" or valor_cat == "Alto"):
            perfil = "💎 Clientes Valiosos"
            acao = "Ofertas personalizadas, cross-sell, up-sell"
        elif recencia_cat == "Alta" and (frequencia_cat == "Alta" or valor_cat == "Alto"):
            perfil = "⚠️ Clientes em Risco"
            acao = "Campanhas de reativação, ofertas especiais"
        elif recencia_cat == "Baixa" and frequencia_cat == "Baixa" and valor_cat == "Baixo":
            perfil = "🆕 Novos Clientes"
            acao = "Campanhas de engajamento, produtos introdutórios"
        elif recencia_cat == "Alta" and frequencia_cat == "Baixa" and valor_cat == "Baixo":
            perfil = "😴 Clientes Dormentes"
            acao = "Campanhas de reativação com desconto"
        else:
            perfil = "📊 Clientes Regulares"
            acao = "Campanhas de manutenção, newsletters"
        
        interpretacoes[cluster] = {
            'perfil': perfil,
            'acao': acao,
            'recencia': f"{recencia_cat} ({media_recencia:.1f} dias)",
            'frequencia': f"{frequencia_cat} ({media_frequencia:.1f} compras)",
            'valor': f"{valor_cat} (R$ {media_valor:.2f})"
        }
    
    return interpretacoes

def main():
    st.set_page_config(
        page_title='RFV + K-means', 
        layout="wide",
        initial_sidebar_state='expanded'
    )

    st.write("""# RFV + Clusterização K-means

    RFV significa recência, frequência, valor e é utilizado para segmentação de clientes baseado no comportamento 
    de compras dos clientes e agrupa eles em clusters parecidos. Utilizando esse tipo de agrupamento podemos realizar 
    ações de marketing e CRM melhores direcionadas, ajudando assim na personalização do conteúdo e até a retenção de clientes.

    Além da segmentação tradicional por quartis, implementamos também **clusterização K-means** para uma segmentação 
    mais sofisticada baseada em machine learning.

    Para cada cliente é preciso calcular cada uma das componentes abaixo:

    - Recência (R): Quantidade de dias desde a última compra.
    - Frequência (F): Quantidade total de compras no período.
    - Valor (V): Total de dinheiro gasto nas compras do período.

    E é isso que iremos fazer abaixo.
    """)
    st.markdown("---")
    
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank marketing data", type=['csv', 'xlsx'])

    if data_file_1 is not None:
        try:
            if data_file_1.name.endswith('.csv'):
                df_compras = pd.read_csv(data_file_1, parse_dates=['DiaCompra'])
            else:
                df_compras = pd.read_excel(data_file_1, parse_dates=['DiaCompra'])
            
            colunas_necessarias = ['DiaCompra', 'ID_cliente', 'CodigoCompra', 'ValorTotal']
            colunas_faltantes = [col for col in colunas_necessarias if col not in df_compras.columns]
            
            if colunas_faltantes:
                st.error(f"❌ Colunas obrigatórias faltantes: {', '.join(colunas_faltantes)}")
                st.info("Colunas necessárias: DiaCompra, ID_cliente, CodigoCompra, ValorTotal")
                return
            
            st.success("✅ Arquivo carregado com sucesso!")
            st.write(f"📊 Total de registros: {len(df_compras)}")
            
            with st.expander("👀 Visualizar primeiras linhas dos dados"):
                st.dataframe(df_compras.head())

            st.write('## Recência (R)')
            
            dia_atual = df_compras['DiaCompra'].max()
            st.write('Dia máximo na base de dados: ', dia_atual)
            st.write('Quantos dias faz que o cliente fez a sua última compra?')

            df_recencia = df_compras.groupby(by='ID_cliente', as_index=False)['DiaCompra'].max()
            df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']
            df_recencia['Recencia'] = df_recencia['DiaUltimaCompra'].apply(lambda x: (dia_atual - x).days)
            st.write(df_recencia.head())

            df_recencia.drop('DiaUltimaCompra', axis=1, inplace=True)

            st.write('## Frequência (F)')
            st.write('Quantas vezes cada cliente comprou com a gente?')
            df_frequencia = df_compras[['ID_cliente', 'CodigoCompra']].groupby('ID_cliente').count().reset_index()
            df_frequencia.columns = ['ID_cliente', 'Frequencia']
            st.write(df_frequencia.head())

            st.write('## Valor (V)')
            st.write('Quanto que cada cliente gastou no periodo?')
            df_valor = df_compras[['ID_cliente', 'ValorTotal']].groupby('ID_cliente').sum().reset_index()
            df_valor.columns = ['ID_cliente', 'Valor']
            st.write(df_valor.head())

            st.write('## Tabela RFV final')
            df_RF = df_recencia.merge(df_frequencia, on='ID_cliente')
            df_RFV = df_RF.merge(df_valor, on='ID_cliente')
            df_RFV.set_index('ID_cliente', inplace=True)
            st.write(df_RFV.head())

            # ========== SEÇÃO K-MEANS ==========
            st.markdown("---")
            st.write('# 🤖 Clusterização K-means')
            st.write('Agora vamos aplicar o algoritmo K-means para uma segmentação mais sofisticada dos clientes.')
            
            # Preparar dados para K-means
            df_for_clustering = df_RFV[['Recencia', 'Frequencia', 'Valor']].copy()
            
            # Padronizar os dados
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_for_clustering)
            
            # Encontrar k ótimo
            st.write('## 📊 Determinando o número ótimo de clusters')
            
            col1, col2 = st.columns(2)
            
            max_clusters = min(10, len(df_RFV) // 2)  # Evitar erro com poucos dados
            k_range, inertias, silhouette_scores = encontrar_k_otimo(df_scaled, max_clusters)
            
            with col1:
                # Gráfico do cotovelo
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(k_range), 
                    y=inertias,
                    mode='lines+markers',
                    name='Inércia',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                fig_elbow.update_layout(
                    title='Método do Cotovelo',
                    xaxis_title='Número de Clusters (k)',
                    yaxis_title='Inércia',
                    template='plotly_white'
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                # Gráfico Silhouette Score
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=list(k_range), 
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
                fig_sil.update_layout(
                    title='Silhouette Score',
                    xaxis_title='Número de Clusters (k)',
                    yaxis_title='Silhouette Score',
                    template='plotly_white'
                )
                st.plotly_chart(fig_sil, use_container_width=True)
            
            # Sugerir k ótimo
            k_otimo_silhouette = k_range[np.argmax(silhouette_scores)]
            st.info(f"💡 **Sugestão baseada em Silhouette Score:** k = {k_otimo_silhouette}")
            
            # Permitir escolha do usuário
            k_escolhido = st.selectbox(
                'Escolha o número de clusters:', 
                options=list(k_range), 
                index=list(k_range).index(k_otimo_silhouette)
            )
            
            if st.button('🚀 Executar Clusterização'):
                # Aplicar K-means
                kmeans = KMeans(n_clusters=k_escolhido, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(df_scaled)
                
                # Adicionar clusters ao dataframe
                df_RFV_clusters = df_RFV.copy()
                df_RFV_clusters['Cluster'] = clusters
                
                st.success(f"✅ Clusterização concluída com {k_escolhido} clusters!")
                
                # Estatísticas dos clusters
                st.write('## 📈 Estatísticas dos Clusters')
                
                cluster_stats = df_RFV_clusters.groupby('Cluster').agg({
                    'Recencia': ['mean', 'std', 'count'],
                    'Frequencia': ['mean', 'std'],
                    'Valor': ['mean', 'std']
                }).round(2)
                
                st.dataframe(cluster_stats)
                
                # Interpretação dos clusters
                st.write('## 🎯 Interpretação dos Clusters')
                interpretacoes = interpretar_clusters(df_RFV_clusters)
                
                for cluster, info in interpretacoes.items():
                    with st.expander(f"Cluster {cluster}: {info['perfil']}"):
                        st.write(f"**Recência:** {info['recencia']}")
                        st.write(f"**Frequência:** {info['frequencia']}")
                        st.write(f"**Valor:** {info['valor']}")
                        st.write(f"**Ação recomendada:** {info['acao']}")
                        
                        # Quantidade de clientes no cluster
                        qtd_clientes = len(df_RFV_clusters[df_RFV_clusters['Cluster'] == cluster])
                        st.write(f"**Quantidade de clientes:** {qtd_clientes}")
                
                # Visualizações
                st.write('## 📊 Visualizações dos Clusters')
                
                # Gráfico 3D
                fig_3d = px.scatter_3d(
                    df_RFV_clusters.reset_index(), 
                    x='Recencia', 
                    y='Frequencia', 
                    z='Valor',
                    color='Cluster',
                    hover_data=['ID_cliente'],
                    title='Clusters RFV - Visualização 3D',
                    labels={
                        'Recencia': 'Recência (dias)',
                        'Frequencia': 'Frequência (compras)',
                        'Valor': 'Valor (R$)'
                    }
                )
                fig_3d.update_traces(marker=dict(size=5))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Gráficos 2D
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_rf = px.scatter(
                        df_RFV_clusters.reset_index(),
                        x='Recencia',
                        y='Frequencia',
                        color='Cluster',
                        title='Recência vs Frequência'
                    )
                    st.plotly_chart(fig_rf, use_container_width=True)
                
                with col2:
                    fig_rv = px.scatter(
                        df_RFV_clusters.reset_index(),
                        x='Recencia',
                        y='Valor',
                        color='Cluster',
                        title='Recência vs Valor'
                    )
                    st.plotly_chart(fig_rv, use_container_width=True)
                
                with col3:
                    fig_fv = px.scatter(
                        df_RFV_clusters.reset_index(),
                        x='Frequencia',
                        y='Valor',
                        color='Cluster',
                        title='Frequência vs Valor'
                    )
                    st.plotly_chart(fig_fv, use_container_width=True)
                
                # Distribuição dos clusters
                fig_dist = px.histogram(
                    df_RFV_clusters.reset_index(),
                    x='Cluster',
                    title='Distribuição de Clientes por Cluster',
                    text_auto=True
                )
                fig_dist.update_traces(textfont_size=12, textangle=0, textposition="outside")
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Adicionar interpretação ao dataframe
                df_RFV_clusters['Perfil_Cluster'] = df_RFV_clusters['Cluster'].map(
                    lambda x: interpretacoes[x]['perfil']
                )
                df_RFV_clusters['Acao_Recomendada'] = df_RFV_clusters['Cluster'].map(
                    lambda x: interpretacoes[x]['acao']
                )
                
                # Salvar resultados
                st.write('## 💾 Resultados da Clusterização')
                st.dataframe(df_RFV_clusters.head(10))
                
                df_clusters_xlsx = to_excel(df_RFV_clusters.reset_index())
                st.download_button(
                    label='📥 Download Resultados K-means',
                    data=df_clusters_xlsx,
                    file_name='RFV_KMeans_Analysis.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            # ========== SEÇÃO RFV TRADICIONAL ==========
            st.markdown("---")
            st.write('# 📊 Segmentação RFV Tradicional (Quartis)')

            st.write('## Segmentação utilizando o RFV')
            st.write("Um jeito de segmentar os clientes é criando quartis para cada componente do RFV, sendo que o melhor quartil é chamado de 'A', o segundo melhor quartil de 'B', o terceiro melhor de 'C' e o pior de 'D'. O melhor e o pior depende da componente. Por exemplo, quanto menor a recência melhor é o cliente (pois ele comprou com a gente tem pouco tempo) logo o menor quartil seria classificado como 'A', já pra componente frequência a lógica se inverte, ou seja, quanto maior a frequência do cliente comprar com a gente, melhor ele/a é, logo, o maior quartil recebe a letra 'A'.")
            st.write('Se a gente tiver interessado em mais ou menos classes, basta a gente aumentar ou diminuir o número de quantis pra cada componente.')

            st.write('Quartis para o RFV')
            quartis = df_RFV.quantile(q=[0.25, 0.5, 0.75])
            st.write(quartis)

            st.write('Tabela após a criação dos grupos')
            df_RFV['R_quartil'] = df_RFV['Recencia'].apply(recencia_class, args=('Recencia', quartis))
            df_RFV['F_quartil'] = df_RFV['Frequencia'].apply(freq_val_class, args=('Frequencia', quartis))
            df_RFV['V_quartil'] = df_RFV['Valor'].apply(freq_val_class, args=('Valor', quartis))
            df_RFV['RFV_Score'] = (df_RFV.R_quartil + df_RFV.F_quartil + df_RFV.V_quartil)
            st.write(df_RFV.head())

            st.write('Quantidade de clientes por grupos')
            st.write(df_RFV['RFV_Score'].value_counts())

            st.write('#### Clientes com menor recência, maior frequência e maior valor gasto')
            clientes_aaa = df_RFV[df_RFV['RFV_Score'] == 'AAA'].sort_values('Valor', ascending=False)
            if len(clientes_aaa) > 0:
                st.write(clientes_aaa.head(10))
            else:
                st.warning("⚠️ Nenhum cliente com score AAA encontrado")

            st.write('### Ações de marketing/CRM')

            dict_acoes = {
                'AAA': 'Enviar cupons de desconto, Pedir para indicar nosso produto pra algum amigo, Ao lançar um novo produto enviar amostras grátis pra esses.',
                'DDD': 'Churn! clientes que gastaram bem pouco e fizeram poucas compras, fazer nada',
                'DAA': 'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar',
                'CAA': 'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar'
            }

            df_RFV['acoes de marketing/crm'] = df_RFV['RFV_Score'].map(dict_acoes)
            st.write(df_RFV.head())

            df_xlsx = to_excel(df_RFV.reset_index())
            st.download_button(
                label='📥 Download Resultados RFV Tradicional',
                data=df_xlsx,
                file_name='RFV_Traditional_Analysis.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            st.write('Quantidade de clientes por tipo de ação')
            st.write(df_RFV['acoes de marketing/crm'].value_counts(dropna=False))

        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.info("Verifique se o arquivo tem o formato correto e as colunas necessárias.")

if __name__ == '__main__':
    main()
