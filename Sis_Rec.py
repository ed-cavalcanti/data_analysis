import pandas as pd
import numpy as np
import ast

# Importar o arquivo com os filmes e visualizar as primeiras linhas
filmes = pd.read_csv('movies_metadata.csv', low_memory = False)
def generos(df):
    if isinstance(df, str):
        df=ast.literal_eval(df)
        nomes=[item['name']for item in df]
        return nomes
    return []
def pegar_nome(df):
    try:
        companies=ast.literal_eval(df)
        nomes=[company['nomes']for company in companies]
        return nomes
    except(ValueError, TypeError):
        return[]
filmes['genres'] = filmes['genres'].apply(generos)

# Importando o arquivo de avaliações e avaliando as primeiras linhas
avaliacoes = pd.read_csv('ratings.csv')

# Seleciona somente as variaveis que iremos utilizar
filmes = filmes [['id','original_title','original_language','vote_count']]

# Renomeia as variaveis
filmes.rename(columns = {'id':'ID_FILME','original_title':'TITULO','original_language':'LINGUAGEM','vote_count':'QT_AVALIACOES'}, inplace = True)

# Seleciona somente as variaveis que iremos utilizar
avaliacoes = avaliacoes [['userId','movieId','rating']]

# Renomeia as variaveis
avaliacoes.rename(columns = {'userId':'ID_USUARIO','movieId':'ID_FILME','rating':'AVALIACAO'}, inplace = True)

# Verificando se há valores nulos
filmes.isna().sum()

# Como são poucos os valores nulos iremos remover porque não terá impacto nenhum
filmes.dropna(inplace = True)

# Verificando se há valores nulos
filmes.isna().sum()

# Verificando se há valores nulos
avaliacoes.isna().sum()

# Verificando a quantidade de avaliacoes por usuarios
avaliacoes['ID_USUARIO'].value_counts()

# Vamos pegar o ID_USUARIO somente de usuários que fizeram mais de 999 avaliações
qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
y = qt_avaliacoes[qt_avaliacoes].index

# Pegando somente avaliacoes dos usuarios que avaliaram mais de 999 vezes
avaliacoes = avaliacoes[avaliacoes['ID_USUARIO'].isin(y)]



# Vamos usar os filmes que possuem somente uma quantidade de avaliações superior a 999 avaliações
filmes = filmes[filmes['QT_AVALIACOES'] > 999]

# Vamos agrupar e visualizar a quantidade de filmes pela linguagem
filmes_linguagem = filmes['LINGUAGEM'].value_counts()


# Selecionar somente os filmes da linguagem EN (English)
filmes = filmes[filmes['LINGUAGEM'] == 'en']

# Visualizar os tipos de dados das variaveis
filmes.info()

avaliacoes.info()

# Precisamos converter a variavel ID_FILME em inteiro
filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)

filmes.info()

# Concatenando os dataframes
avaliacoes_e_filmes = avaliacoes.merge(filmes, on = 'ID_FILME')

# Verificando se há valores nulos
avaliacoes_e_filmes.isna().sum()



# Vamos descartar os valores duplicados, para que não tenha problemas de termos o mesmo usuário avaliando o mesmo filme
# diversas vezes
avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO','ID_FILME'], inplace = True)
display(avaliacoes_e_filmes)

# Vamos excluir a variavel ID_FILME porque não iremos utiliza-la
del avaliacoes_e_filmes['ID_FILME']

# DataFrame sem a variavel ID_FILME
display(avaliacoes_e_filmes.loc[avaliacoes_e_filmes['AVALIACAO'] ==5.0])

# Agora precisamos fazer um PIVOT. O que queremos é que cada ID_USUARIO seja uma variavel com o respectivo valor de nota
# para cada filme avaliado
filmes_pivot = avaliacoes_e_filmes.pivot_table(columns = 'ID_USUARIO', index = 'TITULO', values = 'AVALIACAO')




# Os valores que são nulos iremos preencher com ZERO
filmes_pivot.fillna(0, inplace = True)


# Vamos importar o csr_matrix do pacote SciPy
# Esse método possibilita criarmos uma matriz sparsa
from scipy.sparse import csr_matrix


# Vamos transformar o nosso dataset em uma matriz sparsa
filmes_sparse = csr_matrix(filmes_pivot)

# Tipo do objeto
type(filmes_sparse)

# Vamos importar o algoritmo KNN do SciKit Learn
from sklearn.neighbors import NearestNeighbors 

# Criando e treinando o modelo preditivo
modelo = NearestNeighbors(algorithm = 'brute')
modelo.fit(filmes_sparse)



## Vamos fazer previsões de sugestões de filmes

# 127 Hours
distances, sugestions = modelo.kneighbors(filmes_pivot.filter(items = ['127 Hours'], axis=0).values.reshape(1, -1))

for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]])  

# Toy Story
distances, sugestions = modelo.kneighbors(filmes_pivot.filter(items = ['Toy Story'], axis=0).values.reshape(1, -1))

for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]])  

# 1408
distances, sugestions = modelo.kneighbors(filmes_pivot.filter(items = ['1408'], axis=0).values.reshape(1, -1))

for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]]) 

# 2 Fast 2 Furious
distances, sugestions = modelo.kneighbors(filmes_pivot.filter(items = ['2 Fast 2 Furious'], axis=0).values.reshape(1, -1))

for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]]) 


