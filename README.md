# Topicos

# Trabajo Practico NLP - Detección de Tópicos y clasificación

* Los archivos necesarios para poder correr la aplicación son: **utils.py** (contiene todas las funciones y utilidades que se invocan), **opensearch_data_model.py** (contiene el modelo de datos que se utiliza para la base de datos vectorial), la carpeta **Data** (que contiene los Stopwords) y esta notebook **TP_NLP.ipynb**.

* Todo la ejecución del código se puede hacer desde esta notebook

* Crear un entorno virtual y activarlo

``` bash
conda create -n topics python=3.11
conda activate topics

pip install -r requirements

```

* Levantar el docker con la base de datos vectorial **Opensearch**

docker pull opensearchproject/opensearch:latest  

docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=PassWord#1234! -e "discovery.type=single-node" --name opensearch-node opensearchproject/opensearch:latest  

docker start opensearch-node

## Breve descripción del código


* Todo el código se puede ejecutar desde esta notebook, ya que se invocan las funciones desde los archivos complementarios.  
  
  

**INICIALIZACIÓN DE LA BASE DE DATOS VECTORIAL**

* La función ***init_opensearch()*** verifica si estan creados los índices "topic" y "aux" del archivo **opensearch_data_model.py**, y en caso de que no estén creados, los inicializa.  


**CARGA DE NOTICIAS**
* Para cargar un batch de noticias, se utiliza la función  ***procesar_batch_de_noticias()*** , que es la que orquesta secuencialmente todo el proceso. Hay que pasarle la fecha a procesar, y luego se va imprimiendo en consola los distintos estadíos del proceso y sus resultados.


* Se llama a la función ***obtener_news_batch()***, que selecciona aleatoriamente y para el día indicado, 2.000 noticias del repositorio proporcionado en Hugging Face para trabajar.  


**VOCABULARIO PROPIO Y GENERACIÓN DEL MODELO**
* La función ***vocabulario_propio()*** genera un listado con los tokens ("entities" y "keywords") correspondientes al batch de noticias. Este vocabulario es el que se va a utilizar en el CountVectorizer del modelo.

* La función ***generar_modelo()*** genera el modelo de tópicos con BERTopic, a partir de sus partes fundamentales:
Embedding model: SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'), UMAP, HDBSCAN,  CountVectorizer, ClassTfidfTransformer, KeyBERTInspired.

* El modelo se entrena, y se informa la cantidad de tópicos encontrados y sus nombres.  


**MERGE DE TÓPICOS INTRADIARIO**

* La función ***similitudes_entre_topicos()*** se encarga de determinar si entre los tópicos generados por el modelo, existen tópicos para mergear de forma intradiaria. Para lograrlo, se llama a la función ***matriz_cosine_sim_topicos()*** para que calcule una matriz de similitud coseno entre todos los tópicos. Utilizando dicha matriz, compara sus valores contra un umbral predefinido, y en caso de que un par de tópicos lo supere, lo informa y los incluye una lista para ser mergeados.

* Si existen tópicos para ser mergeados de forma intradiaria, se fusionan con la función ***"merge_topics"*** de BERTopic.  


**INSERCIÓN DE TÓPICOS EN BD VECTORIAL**
* Previo a la inserción de los tópicos en la BD vectorial, se utilizan las funciones auxiliares ***matriz_cosine_sim_docs_topicos()*** y ***generar_lista_tresholds()***, para calcular la similitud coseno entre embeddings de tópicos y documentos, y para determinar el treshold de cada tópico con sus documentos. Dicho treshold se calcula como el promedio de la similitud coseno de los documentos con el tópico.

* La función ***insertar_topicos_en_bd_vctorial()*** es la encargada de orquestar la inserción de tópicos en la base de datos vectorial. Se diferencia entre el primer batch de noticias que se inserta y el resto, ya que para el primer batch no hay datos para calcular posibles merge de tópicos con dias previos.

* Para la inserción en la BD, se construye iterativamente un objeto "Topic" o "Aux" (especificado en el modelo de datos del archivo opensearch_Data_model.py) para luego insertarlo en opensearch.

* Tratandose de tópicos posteriores al primer batch, los mismos son insertados de forma provisoria en el índice "aux" de opensearch, para trabajar sobre posibles merge de tópicos entre dias.

* La función ***recuperar_topicos_bd_vectorial()*** se encarga de escanear y devolver todos los tópicos existentes en la BD, y la función ***recuperar_topicos_bd_vectorial_aux()*** devuelve los tópicos del día procesado.  


**MERGE DE TÓPICOS ENTRE DIAS**

* Tal cual se explicó en clase, ante un nuevo batch de noticias, si las mismas generan un tópico que ya existe, en lugar de generar un nuevo tópico, se debe mergear con uno ya existente.

* La función ***merge_entre_dias()*** se encarga de evaluar la existencia de tópicos para mergear entre distintos dias. Para ello, mediante un cálculo de distancia coseno, evalúa los pares de tópicos actuales que superan el umbral pre establecido con tópicos de dias previos. 

* De no superar el umbral, el tópico se inserta en el índice "topic" como en el caso anterior.

* En caso de superar el umbral, se informa el par de tópicos a mergear, y para ellos se calcula el listado final de keywords que tendrá y el embedding del tópico mergeado.

* La fecha de creación del tópico mergeado se indica como la del día que se está procesando el batch de noticias. La fecha ***"from_date"*** conserva la del tópico original, y la fecha ***"to_date"*** asume el valor del día que se está procesando. De esta forma, se puede hacer un tracking de los tópicos vigentes, y desde cuando existen.

* Por último, se procede a vaciar el índice "aux", en* donde estaban alojados provisoriamente los tópicos del batch de noticias que se estaba procesando. Para ello se utiliza la función ***eliminar_topicos_aux()***  


**INFERENCIA PARA UNA NOTICIA RANDOM**

* Para generar una noticia cualquiera, se utiliza la función ***obtener_noticia_aleatoria()***. La misma, va a devolver el título y el texto de una noticia elegida al azar, dentro de las noticias disponibles para la fecha que se pasa como argumento.

* La función ***procesar_nueva_noticia()*** se encarga de calcular su embedding, extraer entidades y keywords mediante "Spacy", y hacer análisis de sentimiento mediante el modelo de Hugging Face ***"nlptown/bert-base-multilingual-uncased-sentiment"***. 

* Mediante un query de tipo búsqueda KNN a la BD vectorial, se devuelven los 5 tópicos más cercanos a la noticia evaluada, y sus keywords relevantes.
