import streamlit as st
import pandas as pd
from helpers import carregar_dados, converter_para_csv

movies_data = carregar_dados('data\knn\movies.csv');
ratings_data = carregar_dados('data\knn\\ratings.csv');
pln_data = carregar_dados('data\IMDB_Dataset.csv');

st.header('Bases de dados bruta')
st.write('Utilizamos 3 databases para realização deste estudo, sendo elas:')

st.subheader('Algoritmo de recomendação:')

tab1, tab2 = st.tabs(['Filmes', 'Avaliações'])

with tab1:
    st.write('Contém informações de característica dos filmes')

    st.write(movies_data)

    num_linhas, num_colunas = movies_data.shape

    st.markdown(f'O dataframe possui **{num_linhas:,}** linhas e **{num_colunas}** colunas.')

    st.download_button(
        label='Baixar dados em SCV',
        data=converter_para_csv(movies_data),
        file_name='movies.csv',
        mime='text/csv'
    )

with tab2:
    st.write('Contém avaliações em forma de notas de 0 à 5 atribuidas pelos usuários aos filmes')

    st.write(ratings_data.head(50000))

    num_linhas, num_colunas = ratings_data.shape

    st.markdown(f'O dataframe possui **{num_linhas:,}** linhas** e **{num_colunas}** colunas.')
    st.write('** Estão sendo exibidas apenas as primeiras 50.000 linhas devido limitação de exibição do streamlit')

    st.download_button(
        label='Baixar dados em SCV',
        data=converter_para_csv(ratings_data),
        file_name='ratings.csv',
        mime='text/csv'
    )


st.divider()


st.subheader('Algoritmo de PLN:')

st.write('Contém o texto de review e o sentimento (positivo ou negativo) ligado ao texto')

st.write(pln_data)

num_linhas, num_colunas = pln_data.shape

st.markdown(f'O dataframe possui **{num_linhas:,}** linhas e **{num_colunas}** colunas.')

st.download_button(
    label='Baixar dados em SCV',
    data=converter_para_csv(pln_data),
    file_name='reviews.csv',
    mime='text/csv'
)
