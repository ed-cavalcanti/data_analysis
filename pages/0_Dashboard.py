import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import carregar_dados, get_genres_names, get_names

st.title('Dashboard')
st.subheader('Visualização de insigths dos dados do dataframe')

movies_data = carregar_dados('data\knn\movies.csv');

movies_data['genres'] = movies_data['genres'].apply(get_genres_names)
genres = movies_data['genres'].explode().unique()
genres = [genre for genre in genres if genre is not np.nan]

movies_data['production_companies'] = movies_data['production_companies'].apply(get_names)


st.subheader('Porcentagem de gêneros dos filmes')
st.image('charts\genres_percent.png')

st.divider()

st.subheader('Top 10 filmes por gênero*')
st.markdown('*Com base na média de votos no site IMDb')

selected_genre = st.selectbox('Escolha um gênero:', sorted(genres))

movies_by_gender = movies_data[movies_data['genres'].apply(lambda x: selected_genre in x)]
movies_by_gender.sort_values(by='vote_average', ascending=True)
top_10_by_gender = movies_by_gender.head(10).sort_values(by='vote_average')

fig2, ax2 = plt.subplots()
ax2.barh(top_10_by_gender['original_title'], top_10_by_gender['vote_average'])
ax2.set_xlabel('Nota média')
ax2.set_ylabel('Título do filme')
ax2.set_title(f'Top 10 filmes de {selected_genre}')
st.pyplot(fig2)

st.divider()

st.subheader('10 filmes com maior receita')

top_revenue_movies = movies_data.sort_values(by='revenue', ascending=False).head(10)
fig3, ax3 = plt.subplots()
ax3.barh(top_revenue_movies['original_title'][::-1], top_revenue_movies['revenue'][::-1])
ax3.set_xlabel('Receita (bilhões)')
ax3.set_ylabel('Título do filme')
ax3.set_title('10 filmes com maior receita')
st.pyplot(fig3)

st.divider()

st.subheader('Maiores produtoras de filmes*')
st.markdown('*com base na quantidade de filme com nota maior ou igual a 7.0')

companies = movies_data.explode('production_companies')
filtred_movies = companies[companies['vote_average'] >= 7]

movie_count_by_company = pd.DataFrame(filtred_movies['production_companies'].value_counts())

st.bar_chart(movie_count_by_company.head(10))