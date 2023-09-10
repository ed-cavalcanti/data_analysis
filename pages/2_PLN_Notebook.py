import streamlit as st
import streamlit.components.v1 as components

st.title('Visualização de Notebook Jupyter - PLN')

with open('html_notebooks\\nlp.html', 'r', encoding='utf-8') as arquivo:
    conteudo_html = arquivo.read()

components.html(conteudo_html, height=11000, width=800)