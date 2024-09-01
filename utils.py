from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from datasets import load_dataset
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from opensearchpy.helpers import scan
from opensearch_data_model import Topic, Aux ,TopicKeyword, os_client
import spacy
import warnings
warnings.filterwarnings('ignore')


# Cargar el modelo de Spacy para encontrar entidades y keywords
nlp = spacy.load("es_core_news_sm")  # Modelo de Spacy para entidades y keywords
# Cargar el modelo de análisis de sentimiento
sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Modelo de embeddings utilizado en BERTopic
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


SPANISH_STOPWORDS = list(pd.read_csv('data/spanish_stop_words.csv' )['stopwords'].values)

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def es_plural(palabra):
    """
    Función básica para determinar si una palabra es plural en español.
    Este enfoque se basa en reglas simples y no cubrirá todos los casos.
    """
    if palabra.endswith('es') or palabra.endswith('s'):
        return True
    return False

def singularizar(palabra):
    """
    Función para convertir una palabra plural a su forma singular.
    Este es un enfoque simplificado y puede no ser preciso para todos los casos.
    """
    if palabra.endswith('es'):
        return palabra[:-2]
    elif palabra.endswith('s'):
        return palabra[:-1]
    return palabra

def eliminar_plurales(lista_palabras):
  palabras = lista_palabras
  # Conjuntos para almacenar las palabras singulares y las eliminadas
  palabras_singulares = set()
  revisar = []

  # Diccionario para relacionar plurales con sus singulares
  relacion_plural_singular = {}

  for palabra in palabras:
      # Convertir palabra a minúsculas para comparación
      #palabra_minuscula = palabra.lower()

      # Si es plural, verificamos si su singular está en el conjunto de singulares
      if es_plural(palabra):
          singular = singularizar(palabra)
          if singular in palabras_singulares:
              # Si el singular está en el conjunto, agregamos a la lista de revisión
              revisar.append((singular, palabra))
              continue
      
      # Agregamos la palabra singular al conjunto
      palabras_singulares.add(palabra)

  # Convertir el conjunto de palabras singulares en una lista ordenada
  lista_final = sorted(palabras_singulares)
  return revisar, lista_final


def obtener_news_batch(fecha, batch_size=2000):  
    # Hugginface https://huggingface.com/jganzabalseenka
    date_news = fecha
    path_dataset_file = f"jganzabalseenka/news_{date_news}_24hs"
    dataset = load_dataset(path_dataset_file)
    df_parquet = pd.DataFrame(dataset['train'])
    df_1 = df_parquet.sample(n=int(batch_size)).copy()
    print(f'\nSe obtuvieron {len(df_1)} noticias del día {fecha}')
    return df_1

def vocabulario_propio(data, fecha):
    # ARMADO DEL VOCABULARIO PROPIO
    entities = set(sum([list(e) for e in data['entities'].values], []))
    keywords = set(sum([list(e) for e in data['keywords'].values], []))
    all_tokens =  list(entities.union(keywords))
    all_tokens_sort = sorted(all_tokens, key=str.lower)
    print(f'\nSe generaron {len(all_tokens_sort)} tokens de vocabulario propio con noticias del día {fecha}')
    return all_tokens_sort


def generar_modelo(vocabulario, stopwords):
    # COMPONENTES DEL MODELO
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine',
        random_state=100
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True 
    )
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        vocabulary=vocabulario,
        ngram_range=(1, 3)
    )
    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired()

    # MODELO
    topic_model = BERTopic(
    embedding_model=embedding_model,           # Step 1 - Extract embeddings
    umap_model=umap_model,                     # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,               # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,         # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                 # Step 5 - Extract topic words
    representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
    verbose=True
    )
    print(f'\nModelo de tópicos generado')
    return topic_model


def matriz_cosine_sim_topicos(modelo):
    # Función que genera una lista de tópicos y similitud coseno entre ellos 
    # (exceptuando el tópico "-1")
    topicos = [topic for topic in modelo.get_topics().keys() if topic>-1]
    topic_similarities = cosine_similarity(modelo.topic_embeddings_[1:])
    return topicos, topic_similarities


def matriz_cosine_sim_docs_topicos(modelo, documentos):
    # Función que genera una matriz de similitud coseno entre docs y topicos
    # INCLUYE EL TÓPICO "-1"
    print(f'\nGenerando matriz de similitud coseno entre documento y tópicos')
    docs_embeddings = modelo.embedding_model.embed(documentos)
    sim_matrix = cosine_similarity(modelo.topic_embeddings_,docs_embeddings)
    return sim_matrix


def generar_lista_tresholds(modelo, documentos):
    # Función que devuelve una lista con el treshold para cada tópico
    print(f'\nGenerando listado de tresholds para tópicos')

    topicos = [topic for topic in modelo.get_topics().keys() if topic>-1]
    df_topics = pd.DataFrame({'topic': modelo.topics_, 'document':documentos})

    lista_treshold = []
    for i in range(len(topicos)):
        try:
            embedding_del_topico = modelo.topic_embeddings_[1:][i]
            topic_emb_reshaped = embedding_del_topico.reshape(1, -1) 

            filt = df_topics['topic']==i
            documentos_del_topico = list(df_topics.loc[filt]['document'])
            embedding_de_docs = modelo.embedding_model.embed(documentos_del_topico)

            sim_matrix = cosine_similarity(topic_emb_reshaped, embedding_de_docs)
            lista_treshold.append(sim_matrix.mean())
        except:
            lista_treshold.append(0.7)
    return lista_treshold


def get_topic_name(keywords):
    return ', '.join([k for k, s in keywords[:4]])


def get_topic_name_2(keywords):
    return ', '.join([item['name'] for item in keywords[:4]])


def similitudes_entre_topicos(modelo):
    # DEVUELVE LOS TÓPICOS QUE TIENEN SIMILITUD MAYOR AL UMBRAL PARA MERGEAR
    topicos, topic_similarities = matriz_cosine_sim_topicos(modelo)
    umbral = 0.9
    compara_topicos = []
    topicos_similares_id = []
    for i in range(len(topicos)):
        for j in range(i + 1, len(topicos)):
            topic1 = topicos[i]
            topic2 = topicos[j]
            similarity = topic_similarities[i, j]
            compara_topicos.append((topic1, topic2, similarity))
            print(f"La similitud coseno entre el Tópico {topic1} y el Tópico {topic2} es {similarity:.4f}")
            if similarity > umbral:
                topicos_similares_id.append([topic1, topic2])
    if topicos_similares_id:
        print("\nTópicos con similitud coseno mayor al umbral:")
        for topic_1, topic_2 in topicos_similares_id:
            print(f"{topic_1} y {topic_2}")
    return topicos_similares_id


def insertar_topicos_en_bd_vctorial(modelo, sim_matrix, tresholds, documentos, to_date, primer_dia=False):
    print(f'\nInsertando tópicos en base de datos vectorial')
    for topic in modelo.get_topics().keys():
        if topic > -1:
            print(topic)
            keywords = modelo.topic_representations_[topic]
            topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords]
            
            best_doc_index = sim_matrix[topic + 1].argmax()
            best_doc = documentos.iloc[best_doc_index]['title']

            treshold = tresholds[topic]
            
            if primer_dia:
                # ARMADO DEL Topic
                topic_doc = Topic(
                    vector = list(modelo.topic_embeddings_[topic + 1]),
                    similarity_threshold = treshold,
                    created_at = datetime.now(),
                    to_date = parser.parse(to_date),
                    from_date = parser.parse(to_date),
                    index = topic,
                    keywords = topic_keywords,
                    name = get_topic_name(keywords=keywords),
                    best_doc = best_doc
                )
                print(topic_doc.save())
                print()
            else:
                # ARMADO DEL Topic
                topic_doc = Aux(
                    vector = list(modelo.topic_embeddings_[topic + 1]),
                    similarity_threshold = treshold,
                    created_at = datetime.now(),
                    to_date = parser.parse(to_date),
                    from_date = parser.parse(to_date),
                    index = topic,
                    keywords = topic_keywords,
                    name = get_topic_name(keywords=keywords),
                    best_doc = best_doc
                )
                print(topic_doc.save())
                print()


def recuperar_topicos_bd_vectorial():
    # Recuperar todos los tópicos y embeddings de dias anteriores desde OpenSearch
    topics = []
    embeddings = []

    for topic in scan(os_client, index="topic"):
        topics.append(topic)
        embeddings.append(topic['_source']['vector'])
        
    # Convertir a formato numpy array para la similitud coseno
    embeddings = np.array(embeddings)
    return topics, embeddings


def recuperar_topicos_bd_vectorial_aux(fecha):
    # Recuperar los tópicos del día desde OpenSearch
    topics = []
    embeddings = []

    query = {
        "size": 1000,
        "query": {
            "range": {
                "from_date": {
                    "gte": fecha, 
                    "lte": fecha  
                }
            }
        }
    }
 
    # Escaneo de todos los tópicos del día
    for topic in scan(os_client, index="aux", query=query):
        topics.append(topic)
        embeddings.append(topic['_source']['vector'])
        
    # Convertir a formato numpy array para la similitud coseno
    embeddings = np.array(embeddings)
    return topics, embeddings


def topicos_vigentes(fecha):
    # Recuperar los tópicos por fecha "to_date" desde OpenSearch
    query = {
        "size": 1000,
        "query": {
            "range": {
                "to_date": {
                    "gte": fecha, 
                    "lte": fecha  
                }
            }
        }
    }
 
    response = os_client.search(index="topic", body=query)
    df_hits = pd.DataFrame(response['hits']['hits'])
    print(f'Tópicos vigentes a la fecha {fecha}')
    return df_hits


def merge_entre_dias(day1_topics, day1_embeddings, day2_topics, day2_embeddings):
    # Función que realiza (de corresponder) el merge de tópicos nuevos con los del
    # día anterior
    cos_sim = cosine_similarity(day1_embeddings, day2_embeddings)
    threshold = 0.94
    # Recorrer la matriz de similitud coseno y fusionar los tópicos si son similares
    for j, sim_row in enumerate(cos_sim.T):  # Iteramos sobre las columnas
        max_sim = max(sim_row)
        if max_sim > threshold:
            # Índice del tópico del día anterior con mayor similitud
            i = np.argmax(sim_row)

            # Fusionar los embeddings (promedio de vectores)
            merged_embedding = np.mean([day1_embeddings[i], day2_embeddings[j]], axis=0)

            # Fusionar los keywords (evitar duplicados)
            keywords_day1_dict = {kw['name']: kw['score'] for kw in day1_topics[i]['_source']['keywords']}
            keywords_day2_dict = {kw['name']: kw['score'] for kw in day2_topics[j]['_source']['keywords']}
            
            merged_keywords_dict = {**keywords_day1_dict, **keywords_day2_dict}
            merged_keywords = [{'name': name, 'score': score} for name, score in merged_keywords_dict.items()]
            
            # ARMADO DEL Topic
            topic_doc = Topic(
                vector = merged_embedding.tolist(),
                similarity_threshold = day1_topics[i]['_source']['similarity_threshold'],
                created_at = day2_topics[j]['_source']['created_at'],
                to_date = day2_topics[j]['_source']['to_date'],
                from_date = day1_topics[i]['_source']['from_date'],
                index = day2_topics[j]['_source']['index'],
                keywords = merged_keywords,
                name = get_topic_name_2(keywords=merged_keywords),
                best_doc = day1_topics[i]['_source']['best_doc']
            )
                        # Mensaje de fusión
            print(f"Se fusionará el tópico <{day1_topics[i]['_source']['name']}> del día {parser.parse(topic_doc.from_date).strftime('%Y-%m-%d')}\n con el tópico <{day2_topics[j]['_source']['name']}> del día {parser.parse(topic_doc.to_date).strftime('%Y-%m-%d')}.\n Similitud coseno mayor a {threshold}.")
            print()
            print(topic_doc.save())

        else:
            topic_doc = Topic(
                vector = day2_topics[j]['_source']['vector'],
                similarity_threshold = day2_topics[j]['_source']['similarity_threshold'],
                created_at = day2_topics[j]['_source']['created_at'],
                to_date = day2_topics[j]['_source']['to_date'],
                from_date = day2_topics[j]['_source']['from_date'],
                index = day2_topics[j]['_source']['index'],
                keywords = day2_topics[j]['_source']['keywords'],
                name = get_topic_name_2(day2_topics[j]['_source']['keywords']),
                best_doc = day2_topics[j]['_source']['best_doc']
            )
            print(topic_doc.save())


def eliminar_topicos_aux(fecha):
    # Definir el rango de fechas del día 2 para la eliminación
    day2_delete_query = {
        "query": {
            "range": {
                "from_date": {
                    "gte": fecha,
                    "lte": fecha   
                }
            }
        }
    }
    # Eliminar los tópicos originales del día 2
    delete_response = os_client.delete_by_query(index="aux", body=day2_delete_query)
    print(f"Tópicos eliminados: {delete_response['deleted']}")


def procesar_nueva_noticia(nuevo_titulo, nuevo_texto):
    # Función que devuelve tópicos similares, keywords, entidades y analisis de sentimiento para una nueva noticia
    # Generar el embedding del nuevo título y texto
    contenido = nuevo_titulo + " " + nuevo_texto
    new_doc_embed = embedding_model.encode([contenido])[0]

    # Entidades y keywords usando Spacy
    doc = nlp(contenido)
    entidades = list(set([ent.text for ent in doc.ents]))  # Quitar duplicados
    keywords = list(set([token.text.lower() for token in doc if not token.is_stop and not token.is_punct]))  # Quitar duplicados y normalizar

    # Análisis de sentimiento usando el modelo de Hugging Face
    analisis_sentimiento = sentiment_analysis(contenido)

    # Buscar los tópicos similares en la base de datos vectorial utilizando KNN
    query = {
        "size": 5,
        "query": {
            "knn": {
                "vector": {
                    "vector": list(new_doc_embed),
                    "k": 1000
                }
            }
        }
    }

    response = os_client.search(index='topic', body=query)

    # Obtener los IDs de los tópicos más similares
    topicos_similares = [hit["_id"] for hit in response['hits']['hits']]
    similitud_scores = [hit["_score"] for hit in response['hits']['hits']]
    similitud_keywords = [hit['_source']['keywords'] for hit in response['hits']['hits']]
    keywords_similares = []  

    for item in similitud_keywords[0]:  
        keywords_similares.append({'name': item['name'], 'score': item['score']})  

    resultado = {
        "titulo_documento": nuevo_titulo,
        "texto_documento": nuevo_texto,
        "entidades": entidades,
        "keywords": keywords,
        "analisis_sentimiento": {
            "label": analisis_sentimiento[0]['label'],
            "score": round(analisis_sentimiento[0]['score'], 4)
        },
        "topicos_similares": [
            {"topic_id": topicos_similares[i], "similitud_score": round(similitud_scores[i], 4)}
            for i in range(len(topicos_similares))
        ],
        "keywords_similares": keywords_similares
    }
    return resultado


def obtener_noticia_aleatoria(fecha='2024-07-21'):
    # Función que devuelve una noticia aleatoria para la fecha indicada
    path_dataset_file = f"jganzabalseenka/news_{fecha}_24hs"
    dataset = load_dataset(path_dataset_file)
    df_parquet = pd.DataFrame(dataset['train'])
    df_random_new = df_parquet.sample(n=int(1)).copy()
    nuevo_titulo = df_random_new['title'].to_string()
    nuevo_texto = df_random_new['text'].to_string()
    return nuevo_titulo, nuevo_texto


def procesar_batch_de_noticias(fecha, primer_dia=False):
    data = obtener_news_batch(fecha)
    vocabulario = vocabulario_propio(data, fecha)
    topic_model = generar_modelo(vocabulario, SPANISH_STOPWORDS)

    print(f'\nEntrenando modelo....')
    topic_model.fit(data['title_and_text'])
    print(f'\nEntrenamiento finalizado')
    print(f"\nNúmero de tópicos encontrados: {len(topic_model.get_topic_freq())} (incluye el topico -1)")
    print(f'\nLabels de los tópicos generados:')
    print(topic_model.generate_topic_labels())
    
    print(f'\nEvaluando mergear tópicos del mismo día...')
    topicos_para_merge = similitudes_entre_topicos(topic_model)
    
    if topicos_para_merge:
        topic_model.merge_topics(data['title_and_text'], topicos_para_merge)
        print(f'\nSe realizó el merge de los siguientes tópicos: {topicos_para_merge}')
    else:
        print(f'\nNo existen tópicos para mergear intradía')

    sim_matrix = matriz_cosine_sim_docs_topicos(topic_model ,list(data['title']))
    lista_tresholds = generar_lista_tresholds(topic_model, data['title'])
    insertar_topicos_en_bd_vctorial(topic_model, sim_matrix, lista_tresholds, data, fecha, primer_dia)
    print(f'\nSe insertaron los tópicos en la BD vectorial')

    if primer_dia:
        print(f'\nCarga de batch con noticias finalizado!')
    else:
        print(f'\nEvaluando mergear tópicos entre dias...')
        day1_topics, day1_embeddings = recuperar_topicos_bd_vectorial()
        day2_topics, day2_embeddings = recuperar_topicos_bd_vectorial_aux(fecha)
        merge_entre_dias(day1_topics, day1_embeddings, day2_topics, day2_embeddings)
        print(f'\nLimpiando tabla auxiliar...')
        eliminar_topicos_aux(fecha)
        print(f'\nCarga de batch con noticias finalizado!')

