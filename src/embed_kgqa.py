import tensorflow as tf
from tensorflow import keras


class EmbedKGQA(keras.Model):
    def __init__(self, graph_embedding_model):
        super(EmbedKGQA, self).__init__()
        self.fc1 = keras.layers.Dense(512, input_shape=(768,))
        self.fc2 = keras.layers.Dense(512)
        self.fc3 = keras.layers.Dense(512)
        self.graph_embedding_model = graph_embedding_model
        self.graph_embedding_model.trainable = False

    def call(self, q_embeddings, q_entity_embeddings):
        question_complex = self.fc1(q_embeddings)
        question_complex = self.fc2(question_complex)
        question_complex = self.fc3(question_complex)

        scores = self.graph_embedding_model.get_score(
            q_entity_embeddings,
            question_complex,
            self.graph_embedding_model.entity_encoder
        )

        prediction = tf.sigmoid(scores)
        return prediction