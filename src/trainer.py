import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from src import config
import numpy as np
import os

from src.embed_kgqa import EmbedKGQA
from complex_graph_embedding.model import KBEmbedding
from complex_graph_embedding.data_loader import data_loader
from complex_graph_embedding import config as kb_config
from src.metaqa_loader import MetaQADataLoader


class Trainer:
    def __init__(self, embedKGQA, kb_embedder, loss_fn, optimizer, train_data_loader, validation_data_loader,
                 epochs, target_dim, batch_size, model_name='default', log_iters=config.TRAIN_LOG_ITERATIONS,
                 save_base='../data/saved_models'):
        self.embedKGQA = embedKGQA
        self.kb_embedder = kb_embedder
        self.loss_fn = loss_fn
        self.log_iters = log_iters
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.least_loss = np.inf
        self.train_loss_mean = keras.metrics.Mean(name='train_loss_mean')
        self.validation_loss_mean = keras.metrics.Mean(name='validation_loss_mean')
        self.save_path = os.path.join(save_base, model_name)

    @tf.function
    def train_step(self, q_embeddings, q_entity_embeddings, targets):
        with tf.GradientTape() as tape:
            predictions = self.embedKGQA(
                q_embeddings,
                q_entity_embeddings,
                training=True
            )
            loss = self.loss_fn(y_true=targets, y_pred=predictions)
        grads = tape.gradient(loss, self.embedKGQA.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.embedKGQA.trainable_variables))
        self.train_loss_mean(loss)

    @tf.function
    def validation_step(self, q_embeddings, q_entity_embeddings, targets):
        predictions = self.embedKGQA(
            q_embeddings,
            q_entity_embeddings,
            training=False
        )
        loss = self.loss_fn(y_true=targets, y_pred=predictions)
        self.validation_loss_mean(loss)

    def store_best_model(self, loss):
        if loss < self.least_loss:
            print('validation loss decreased from {} to {}. saving model.'.format(
                self.least_loss, loss)
            )
            self.least_loss = loss
            self.embedKGQA.save_weights(self.save_path)

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            self.train_loss_mean.reset_states()
            self.validation_loss_mean.reset_states()

            iteration = 0
            for _x, _y in self.train_data_loader.get_batch(self.target_dim, self.batch_size):
                iteration += 1
                self.train_step(_x[0], _x[1], _y)

                if not iteration % self.log_iters:
                    print('training loss in iteration {}: {}'.format(iteration, self.train_loss_mean.result()))

            for _x, _y in self.validation_data_loader.get_batch(self.target_dim, config.VALIDATION_BATCH_SIZE):
                self.validation_step(_x[0], _x[1], _y)

            validation_loss = self.validation_loss_mean.result()
            print("epoch:{} validation_loss:{}".format(epoch, validation_loss))
            self.store_best_model(validation_loss)


if __name__ == '__main__':

    kb_embedding_model = KBEmbedding(
        entity_dim=data_loader.entity_dim,
        relation_dim=data_loader.relation_dim,
        hidden_dim=kb_config.HIDDEN_DIMENSION,
        scoring='complex'
    )

    embed_kgqa = EmbedKGQA(kb_embedding_model)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    train_data_loader = MetaQADataLoader(
        kb_mgr=data_loader.kb_mgr,
        paths=['../data/1-hop/qa_train.txt'],
        kb_embedding_model=kb_embedding_model,
        contextual_embedder='roberta'
    )

    validation_data_loader = MetaQADataLoader(
        kb_mgr=data_loader.kb_mgr,
        paths=['../data/1-hop/qa_dev.txt'],
        kb_embedding_model=kb_embedding_model,
        contextual_embedder='roberta'
    )

    trainer = Trainer(
        embedKGQA=embed_kgqa,
        kb_embedder=kb_embedding_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        epochs=config.EPOCHS,
        target_dim=data_loader.target_dim,
        batch_size=config.BATCH_SIZE,
        model_name="embedkgqa_local_gpu",
    )

    trainer.run()
