###########################
######
###### Warranty - WordCount (Embeddings)
#####
###########################
###### Packages
import string
import unicodedata
import pandas as pd 
import re
from datetime import datetime
###########################


def step02_waranty_transform(df):

    # Positives due warranty time specification
    palavras = ['1 ano', '2 anos', '5 anos', '3 meses', '4 meses' , '12 meses']
    regex = "|".join([rf"\b{palavra}\b" for palavra in palavras])
    df["trat_warranty_time"]= df["warranty"].apply(lambda x: '' if x is None else unicodedata.normalize('NFKD', x) \
                                .encode('ASCII', 'ignore').decode('utf-8')) \
                                .str.translate(str.maketrans('', '', string.punctuation)) \
                                .str.contains(regex, regex=True).apply(lambda x: bin(1 if x == True else 0)[2:])

    # Positives due Origin
    palavras = ["fabrica","fabricacion","fabricante","directo","directamente","novo","oficial","nuevo"]
    regex = "|".join([rf"\b{palavra}\b" for palavra in palavras])
    df["trat_warranty_fabric"]= df["warranty"].apply(lambda x: '' if x is None else unicodedata.normalize('NFKD', x) \
                                .encode('ASCII', 'ignore').decode('utf-8')) \
                                .str.translate(str.maketrans('', '', string.punctuation)) \
                                .str.contains(regex, regex=True).apply(lambda x: bin(1 if x == True else 0)[2:])
    
    # Negative due reputation needs check
    palavras = ["reputacion" ,"calificacion","calificaciones","sin garantia","revisan", "revisados", "buen", "buenas", "corresponden","perfecto","articulo","descripcion","revisa","completo"]
    regex = "|".join([rf"\b{palavra}\b" for palavra in palavras])
    df["trat_warranty_reputation"]= df["warranty"].apply(lambda x: '' if x is None else unicodedata.normalize('NFKD', x) \
                                .encode('ASCII', 'ignore').decode('utf-8')) \
                                .str.translate(str.maketrans('', '', string.punctuation)) \
                                .str.contains(regex, regex=True).apply(lambda x: bin(1 if x == True else 0)[2:])
    
    # Negative due image needs
    palavras = ["foto","fotos","imagem","imagenes"]
    regex = "|".join([rf"\b{palavra}\b" for palavra in palavras])
    df["trat_warranty_image"]= df["warranty"].apply(lambda x: '' if x is None else unicodedata.normalize('NFKD', x) \
                                .encode('ASCII', 'ignore').decode('utf-8')) \
                                .str.translate(str.maketrans('', '', string.punctuation)) \
                                .str.contains(regex, regex=True).apply(lambda x: bin(1 if x == True else 0)[2:])
    
    return df