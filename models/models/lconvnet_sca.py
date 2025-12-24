import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout,
    GlobalAveragePooling1D, Flatten, TimeDistributed,
    Reshape, LayerNormalization, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.sca import ScaledCustomAttention


def build_lconvnet_sca(
    n_timesteps=256,
    n_channels=25,
    n_classes=1,
    dropout_rate=0.5
):
    input_tensor = Input(shape=(n_timesteps, n_channels))

    reshaped = Reshape((n_channels, n_timesteps, 1))(input_tensor)

    x = Conv2D(32, (3, 3), padding="same", activation="relu")(reshaped)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (7, 7), padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    td = TimeDistributed(Flatten())(x)
    td = TimeDistributed(Dense(64, activation="relu"))(td)

    sca_out = ScaledCustomAttention()(td)
    sca_out = Add()([td, sca_out])
    sca_out = LayerNormalization()(sca_out)

    lstm_out = LSTM(64, return_sequences=False)(sca_out)
    global_pool = GlobalAveragePooling1D()(td)

    concat = Concatenate()([lstm_out, global_pool])
    output = Dense(n_classes, activation="sigmoid")(concat)

    model = Model(input_tensor, output)
    model.compile(
        optimizer=Adam(learning_rate=1e-5, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
