from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def get_finetuned_mlp(input_shape, output_size=1, output_activation='sigmoid',
                      optimizer=Adam(1e-4),  # "rmsprop",
                      loss="binary_crossentropy", metrics=None):
    """
    Define MLP and compile loss on top. Default settings are for a binary classifier (i.e., gender recognition).
    @param output_size: number of output predictions (i.e., number of classes or just 1 for binary)
    @type output_size: int
    @param output_activation:   activation function used to normalize the output
    @param optimizer:   optimization schema
    @param loss:    type of loss function
    @param metrics:     metrics to optimize according to
    @type metrics: list
    @return: Keras model compiled with loss on top of MLP

    """
    if metrics is None:
        metrics = ["accuracy"]
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_size, activation=output_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
