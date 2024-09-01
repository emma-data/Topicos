from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, DenseVector, Nested, Date, Object, connections, InnerDoc
import os




OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', "localhost")
auth = ('admin', 'PassWord#1234!')
port = 9200
os_client = connections.create_connection(
    hosts = [{'host': OPENSEARCH_HOST, 'port': port}],
    http_auth = auth,
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    alias='default'
    # ssl_assert_hostname = False,
    # ssl_show_warn = False
)

TOPIC_DIMENSIONS = 384
TOPIC_INDEX_NAME = 'topic'
TOPIC_INDEX_PARAMS = {
    'number_of_shards': 1,
    'knn': True
}

AUX_INDEX_NAME = 'aux'
AUX_INDEX_PARAMS = {
    'number_of_shards': 1,
    'knn': True
}

knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}

class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class Topic(Document):
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)
    similarity_threshold = Float()
    created_at = Date()
    to_date = Date()
    from_date = Date()
    index = Integer()
    keywords = Object(TopicKeyword)
    name = Text()
    best_doc = Text()
    
    class Index:
        name = TOPIC_INDEX_NAME
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + '_' + self.name.replace(', ', '-')
        return super(Topic, self).save(** kwargs)



class Aux(Document):
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)
    similarity_threshold = Float()
    created_at = Date()
    to_date = Date()
    from_date = Date()
    index = Integer()
    keywords = Object(TopicKeyword)
    name = Text()
    best_doc = Text()
    
    class Index:
        name = AUX_INDEX_NAME
        if not os_client.indices.exists(index=AUX_INDEX_NAME):
            settings = {
                'index': AUX_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + '_' + self.name.replace(', ', '-')
        return super(Aux, self).save(** kwargs)
    


def init_opensearch():
    """
    Inicializa los índices de la base Opensearch si no fueron creados
    """
    # Verifica si existe el índice Topic
    if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
        Topic.init()
        print("Índice Topic creado")
    else:
        print("El índice Topic ya existe.")

    # Verifica si existe el índice Aux
    if not os_client.indices.exists(index=AUX_INDEX_NAME):
        Aux.init()
        print("Índice Aux creado")
    else:
        print("El índice Aux ya existe.")
