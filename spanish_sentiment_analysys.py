# ! pip install spanish_sentiment_analysis

from classifier import * 

import pandas as pd
import streamlit as st
import time

nlp = SentimentClassifier()

def main():

    st.title('Coke.ai')
    st.title('Análisis de sentimiento ...')

    # text = st.text_input("Expresión:")
    write_here = "Texto aqui..."
    text = st.text_area("Incluya un texto ..", write_here)
    if st.button("Analizar"):
        if text != write_here:
            label, score = sentimiento(text)
            st.success('Sentimiento de ['+ text + ']')
            st.success(label)
            st.success('%.2f' % score)    
        else:
            st.error("Ingresa un texto y presiona el boton Analizar ..")
    else:
        st.info(
            "Ingresa un texto y presiona el boton Analizar .."
        )

    uploaded_file = st.file_uploader("O bien puede seleccionar un archivo CSV para procesar múltiples párrafos (se procesará columna 'text')",type=['csv'])
    if uploaded_file is not None:
        if st.button("Procesar Archivo CSV"):
            data = pd.read_csv(uploaded_file)
            st.success("Procesando CSV ..")
            data[data['text'].str.strip().astype(bool)]
            #indexes = data[data.text_len <  30].index
            #data = data.drop(indexes)
            data['text'] = data['text'].astype(str)
            total_reg = len(data)
            #my_bar = st.progress(0)
            #i = 0
            t0 = time.time()
            msg = f"Espere por favor, esto puede tomar algun tiempo .. procesando {total_reg:.0f} elementos"  if total_reg>1000 else f"Espere .. procesando {total_reg:.0f} elementos"
            with st.spinner(msg):
                g = lambda x: pd.Series(sentimiento(x.text))
                data[['label', 'score']] = data.apply(g, axis=1)
            #sentences = data.text.tolist()
            #i=0
            #
            #
            #for sentence in sentences: # tqdm(sentences):
            #    label, score = sentimiento(sentence)
            #    data.loc[data.index[i], 's5'] = round(score, 4)
            #    data.loc[data.index[i], 's5_label'] = label
            #    i=i+1
            #    my_bar.progress(i/total_reg)
            #indexes = data[data.sentiment >=  0].index
            #data = data.drop(indexes)
            #data.to_csv("/Users/RDPulgar/Google Drive/AI/cv/P095_HT/_cokeai_results.csv")
            csv = convert_df(data)
            st.success(f'Procesado con éxito en {time.time() - t0:.0f} seg')
            if st.download_button(label="Presione para descargar archivo procesado", data=csv, file_name='_cokeai_results.csv', mime='text/csv'):
                st.success("Descargado con éxito ..")
                st.stop()
        else:
            st.error("Aun no se ha procesado el archivo")
    else:
        st.info("Cargando el archivo ..")


def CheckForLess(list1, val): 
      
    # traverse in the list
    i=1
    for x in list1: 
          if val <= x: 
            return i
          else:
            i=i+1
    return False

def sentimiento(text):
    #try:
        conditions = {
            1: 'Muy Malo',
            2: 'Malo',
            3: 'Neutro',
            4: 'Bueno',
            5: 'Muy bueno'
        }
        result = nlp.predict(text)
        label = conditions[CheckForLess([0.1, 0.2, 0.5, 0.8, 1],result)]
        return label, round(result,4)
    #except:
    #    return "_Error", -1

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if __name__ == '__main__':
    main()