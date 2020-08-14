import tensorflow as tf
import numpy as np
from complex_graph_embedding import config
import os

from tqdm import tqdm
from tensorflow import keras
from complex_graph_embedding.model import KBEmbedding
from complex_graph_embedding.data_loader import data_loader


class Trainer:
    def __init__(self, model, optimizer, loss_fn, data_loader, epochs,
                 batch_size, name='default'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loss_mean = keras.metrics.Mean(name='train_loss_mean')
        self.validation_loss_mean = keras.metrics.Mean(name='validation_loss_mean')
        self.least_loss = np.inf
        self.save_path = os.path.join(config.SAVED_MODELS_PATH, name)

    @tf.function
    def train_step(self, subj_ids, rel_ids, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(
                subj_ids,
                rel_ids,
                training=True
            )
            loss = self.loss_fn(y_true=targets, y_pred=predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_mean(loss)

    @tf.function
    def validation_step(self, subj_ids, rel_ids, targets):
        predictions = self.model(
            subj_ids,
            rel_ids,
            training=False
        )
        loss = self.loss_fn(y_true=targets, y_pred=predictions)
        self.validation_loss_mean(loss)

    def evaluate(self):
        print("Loading best model.")

        precision = []
        recall = []

        self.model.load_weights(self.save_path)
        for x, y in tqdm(self.data_loader.get_batch('test', config.TEST_BATCH_SIZE)):
            subj_ids, rel_ids = list(zip(x))
            predictions = self.model(
                np.array(subj_ids),
                np.array(rel_ids),
                training=False
            )
            test_loss = self.loss_fn(y_true=y, y_pred=predictions)

            normalized_pred = np.where(predictions > config.PREDICTION_THRESHOLD, 1.0, 0.0)
            tp = len(normalized_pred[(normalized_pred == y) & (normalized_pred == 1.0)])
            p = len(y[y == 1.0])
            fp = len(normalized_pred[(normalized_pred != y) & (normalized_pred == 1.0)])
            print(tp, p, fp)
            recall.append(tp/(p+0.001))
            precision.append(tp/(tp+fp+0.001))

        print('model test loss:', test_loss)
        print('precision: {:.3f},  recall:{:.3f}'.format(np.mean(precision), np.mean(recall)))

    def store_best_model(self, loss):
        if loss < self.least_loss:
            print('validation loss decreased from {} to {}. saving model.'.format(
                self.least_loss, loss)
            )
            self.least_loss = loss
            self.model.save_weights(self.save_path)

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            self.train_loss_mean.reset_states()
            self.validation_loss_mean.reset_states()

            iteration = 0
            for _x, _y in self.data_loader.get_batch('train', self.batch_size):
                iteration += 1
                subj_ids, rel_ids = list(zip(_x))
                self.train_step(np.array(subj_ids), np.array(rel_ids), _y)

                if not iteration % config.TRAIN_LOG_ITERATIONS:
                    print('training loss in iteration {}: {}'.format(iteration, self.train_loss_mean.result()))

            # for _x, _y in self.data_loader.get_batch('validation', self.batch_size):
            #     subj_ids, rel_ids = list(zip(_x))
            #     self.validation_step(np.array(subj_ids), np.array(rel_ids), _y)

            validation_loss = self.train_loss_mean.result()
            # validation_loss = self.validation_loss_mean.result()
            print("epoch:{} validation_loss:{}".format(epoch, validation_loss))
            self.store_best_model(validation_loss)


if __name__ == '__main__':
    model = KBEmbedding(
        entity_dim=data_loader.entity_dim,
        relation_dim=data_loader.relation_dim,
        hidden_dim=config.HIDDEN_DIMENSION,
        scoring='complex'
    )

    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=data_loader,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        name='complex'
    )

    trainer.run()
    # trainer.evaluate()