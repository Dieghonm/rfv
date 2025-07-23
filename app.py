import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO

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
def main():
    st.set_page_config(
        page_title='RFV', 
        layout="wide",
        initial_sidebar_state='expanded'
    )

    st.write("""# RFV

    RFV significa recência, frequência, valor e é utilizado para segmentação de clientes baseado no comportamento 
    de compras dos clientes e agrupa eles em clusters parecidos. Utilizando esse tipo de agrupamento podemos realizar 
    ações de marketing e CRM melhores direcionadas, ajudando assim na personalização do conteúdo e até a retenção de clientes.

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

            df_xlsx = to_excel(df_RFV)
            st.download_button(
                label='📥 Download Resultados RFV',
                data=df_xlsx,
                file_name='RFV_Analysis.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            st.write('Quantidade de clientes por tipo de ação')
            st.write(df_RFV['acoes de marketing/crm'].value_counts(dropna=False))

        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.info("Verifique se o arquivo tem o formato correto e as colunas necessárias.")

if __name__ == '__main__':
    main()