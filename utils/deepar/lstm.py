from main.config import get_param
from utils.deepar import NNModel
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from utils.deepar.loss import quantile_mae
import logging

logger = logging.getLogger('deepar')


class DeepAR(NNModel):
    def __init__(self, train_gen, val_gen, val_steps, steps_per_epoch, epochs, loss=quantile_mae, optimizer='adam'):
        super().__init__()
        self.train_gen = train_gen
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.val_gen = val_gen
        self.val_steps = val_steps
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model = None
        self.nn_structure = DeepAR.basic_structure

    @staticmethod
    def basic_structure():
        """
        This is the method that needs to be patched when changing NN structure
        :return: input (Tensor), output (Tensor)
        """
        input = Input(shape=(get_param('lookback'), 1))
        x = LSTM(get_param('unit'), return_sequences=True)(input)
        x = Lambda(lambda a: K.reshape(a, (-1, get_param('unit') * get_param('lookback'))))(x)
        output = Dense(1)(x)
        return input, output

    def instantiate_and_fit(self, q, verbose=False):
        input, output = self.nn_structure()
        model = Model(input, output)
        model.compile(loss=self.loss(q), optimizer=self.optimizer)
        history = model.fit(self.train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            validation_data=self.val_gen,
                            validation_steps=self.val_steps)
        # self._plot_learning_curves(history)
        if verbose:
            logger.debug('Model was successfully trained')
        self.keras_model = model

    # def _plot_learning_curves(self, history):
    #     """
    #     :param history:
    #     :return:
    #     """
    #     pd.DataFrame(history.history).plot(figsize=(8, 5))
    #     plt.grid(True)
    #     plt.gca().set_ylim(0, 1)
    #     plt.show()

    @property
    def model(self):
        return self.keras_model

    def predict(self, test_X):
        return self.keras_model.predict(test_X)
