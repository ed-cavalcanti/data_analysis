import pandas as pd
import streamlit as st
import numpy as np
import ast

genres_to_remove = ['The cartel', 'Rogue State', 'Pulser Productions', 'Odyssey Media', 'Sentai Filmworks', 'Mardock Scramble Production Committee', 'BROSTA TV', 'Aniplex', 'Telescene Film Group Productions', 'Vision View Entertainment', 'Carousel Productions', 'GoHands','The Cartel', np.nan]

@st.cache_data
def carregar_dados(path: str):
    dados = pd.read_csv(path, lineterminator='\n')
    return dados;

@st.cache_data
def converter_para_csv(dataframe: pd.DataFrame):
    return dataframe.to_csv().encode('utf-8')

def get_genres_names(data: str) -> list:
    if isinstance(data, str):
        data = ast.literal_eval(data)
        names = [item['name'] for item in data if item['name'] not in genres_to_remove]
        # names = remover_valores(names)
        return names
    return []

def get_names(data: str) -> list:
    try:
        companies = ast.literal_eval(data)
        names = [company['name'] for company in companies]
        return names
    except (ValueError, TypeError):
        return []

def remover_nulos(lista):
    return [genero for genero in lista if genero is not np.nan]
