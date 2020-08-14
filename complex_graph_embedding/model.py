import tensorflow as tf
from tensorflow import keras


class KBEmbedding(keras.Model):

    def __init__(self, entity_dim, relation_dim, hidden_dim, scoring):
        super(KBEmbedding, self).__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim

        if scoring == 'distmult':
            self.get_score = self.dist_mult
        elif scoring == 'complex':
            self.get_score = self.complEx
        else:
            raise NotImplementedError()

        self.entity_encoder = keras.layers.Embedding(
            self.entity_dim,
            self.hidden_dim,
            embeddings_regularizer=keras.regularizers.l2(0.1)
        )

        self.relation_encoder = keras.layers.Embedding(
            self.relation_dim,
            self.hidden_dim,
            input_shape=(),
        )

        self.head_bn = keras.layers.BatchNormalization()
        self.head_drpout = keras.layers.Dropout(0.3)
        self.rel_drpout = keras.layers.Dropout(0.4)
        self.score_bn = keras.layers.BatchNormalization()
        self.output_drpout = keras.layers.Dropout(0.5)

    def dist_mult(self, head, relation, entity_encoder):
        """

        :param head: the head entity or the subject
        :param relation:
        :param entity_encoder: the entity encoder layer
        :return:
        """

        head_norm = self.head_bn(tf.squeeze(head))
        head_drp = self.head_drpout(head_norm)
        relation_drp = self.rel_drpout(tf.squeeze(relation))

        scores = tf.math.multiply(head_drp, relation_drp)
        scores_norm = self.score_bn(scores)
        scores_drp = self.output_drpout(scores_norm)
        scores = tf.matmul(
            scores_drp,
            tf.squeeze(entity_encoder.weights),
            transpose_b=True
        )

        return tf.squeeze(scores)

    def complEx(self, head, relation, entity_encoder):
        """

        :param head: the head entity or the subject
        :param relation:
        :param entity_encoder: the entity encoder layer
        :return:
        """
        assert self.hidden_dim % 2 == 0
        hidden_dim_slice = int(self.hidden_dim/2)

        head_norm = self.head_bn(tf.reshape(head, (-1, hidden_dim_slice, 2)))
        head_drp = self.head_drpout(head_norm)

        head_drp = tf.reshape(head_drp, (-1, self.hidden_dim))

        re_head = tf.slice(head_drp, [0, 0], [-1, hidden_dim_slice])
        im_head = tf.slice(head_drp, [0, hidden_dim_slice], [-1, -1])

        relation_drp = self.rel_drpout(tf.squeeze(relation))
        re_relation = tf.slice(relation_drp, [0, 0], [-1, hidden_dim_slice])
        im_relation = tf.slice(relation_drp, [0, hidden_dim_slice], [-1, -1])

        re_tail = tf.slice(tf.squeeze(entity_encoder.weights), [0, 0], [-1, hidden_dim_slice])
        im_tail = tf.slice(tf.squeeze(entity_encoder.weights), [0, hidden_dim_slice], [-1, -1])

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = tf.stack([re_score, im_score], axis=1)
        score_bn = self.score_bn(score)
        score_drp = self.output_drpout(score_bn)

        score_drp = tf.reshape(score_drp, (-1, self.hidden_dim))
        re_score = tf.slice(score_drp, [0, 0], [-1, hidden_dim_slice])
        im_score = tf.slice(score_drp, [0, hidden_dim_slice], [-1, -1])

        scores = tf.add(
            tf.matmul(re_score, re_tail, transpose_b=True),
            tf.matmul(im_score, im_tail, transpose_b=True)
        )

        return scores

    def call(self, subj_ids, rel_ids):
        entity_embedding = self.entity_encoder(subj_ids)
        rel_embedding = self.relation_encoder(rel_ids)

        scores = self.get_score(entity_embedding, rel_embedding, self.entity_encoder)
        prediction = tf.sigmoid(scores)

        return prediction


