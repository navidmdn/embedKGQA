KB_PATH = './data/kb.txt'
ENTITY_VOCABULARY_PATH = './data/entity_vocab.pickle'
RELATION_VOCABULARY_PATH = './data/relation_vocab.pickle'


# training configs

TEST_SPLIT = 0.00001
VALIDATION_SPLIT = 0.00001
TRAIN_LOG_ITERATIONS = 100
EPOCHS = 50
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.001


# model configs
HIDDEN_DIMENSION = 512
SAVED_MODELS_PATH = 'data/saved_models'

# evaluation configs
PREDICTION_THRESHOLD = 0.03