import streamlit as st 
import joblib
import spacy
import pandas as pd 

#configuração da página
st.set_page_config(page_title="Triagem de chamados", page_icon="🤖")

#carregamento de recursos -> em cache para que não seja necessário recarregar a cada clique
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_triagem_suporte.pkl") #carregamento do modelo de ML treinado


@st.cache_resource
def carregar_nlp():
    return spacy.load("pt_core_news_sm") #carregamento do modelo de ML para língua portuguesa 
    
try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()
    
except:
    st.error("Erro: Execute o script 'treinar_modelo_py' para gerar o aquivo .pkl")
    st.stop()
    
#lógica de processamento
def analisar_chamado(texto_usuario):
    
    #1.processamento linguístico com spacy
    doc = nlp(texto_usuario)
    
    #extração de entidades nomeadas (ex.: AWS, locais, equipamento e etc.)
    entidades = [(ent.text,ent.label_) for ent in doc.ents]
    
    #limpeza de texto pro modelo: lematizar, converter ´ra minúsculo e remover pontuação
    texto_limpo = " ".join([
        token.lemma_.lower()
        for token in doc 
        if not token.is_punct
    ])

    #2. predição com o modelo de machine learning

    #prevê a categoria do chamado
    categoria_predita = modelo.predict([texto_limpo])[0]

    #probabilidade de cada categoria / classe
    probs = modelo.predict_proba([texto_limpo])[0]

    #pega a maior probabilidade como resultado da análise
    confianca = max(probs)*100

    #retorna os resultados
    return categoria_predita, confianca, entidades

#-------------interface gráfica--------------#
st.title("Triagem de suporte") #título da página
st.markdown("Decreva o problema em poucas palavras.") # descrição

#criando o chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    
#exibe mensagens anteriores no chat
for message in st.session_state.messages: #criar um armazenamento pra cada mensagem
    with st.chat_message(message["role"]): #criar um botão de chat virtual
        st.markdown(message["content"]) #exibir o texto da mensagem dentro do balão
        
#exibir o campo de entrada para o usuário (chat input)
if prompt:= st.chat_input("Ex.: O servidor AWS parou de responder..."):
    
    st.chat_message('user').markdown(prompt)
    
    st.session_state.messages.append({
        "role":"user",
        "content": prompt
    })     
    
#processar a resposta que a IA (modelo de ML) vai retornar ao usuário
    categoria, confianca, ents = analisar_chamado(prompt)

#montar/personalizar a resposta em um formato amigável
    resposta_md = f"""
**Análise do chamado:**
**Categoria:** `{categoria}`
**Confiança:** `{confianca:.2f}%`
"""
    
    if ents:
        resposta_md += "\n\n **Entidades detectadas:**"
        for ent in ents:
            resposta_md += f"\n-*{ent[0]}*({ent[1]})"
                
    #ações automáticas por categoria
    acoes = {
        "Infraestrutura": "Encaminhando para equipe N2",
        "Acesso": "Verificando logs de autenticação.",
        "Hardware": "Abrindo ordem de serviço",
        "Software": "Verificando disponibilidade de licenças."
    }            

#adicionar as açoes sugeridas com base na categoria
    resposta_md += f"\n\n **Ação:** {acoes.get(categoria, 'Triagem manual necessária.')}"

#3. exibir a resposta do assistente

    with st.chat_message('assistant'):
        st.markdown(resposta_md)
    
#salvar resposta no histórico
    st.session_state.messages.append({
        "role":"assistant",
        "content": resposta_md
    })
