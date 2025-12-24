from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout,
    GlobalAveragePooling1D, Flatten, TimeDistributed,
    Reshape, LayerNormalization, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .sca import ScaledCustomAttention

def build_lconvnet_sca(
    n_timesteps=256,
    n_channels=25,
    dropout_rate=0.5
):
    input_tensor = Input(shape=(n_timesteps, n_channels))
    x = Reshape((n_channels, n_timesteps, 1))(input_tensor)

    x = Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (7,7), padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(x)

    td = TimeDistributed(Flatten())(x)
    td = TimeDistributed(Dense(64, activation="relu"))(td)

    att = ScaledCustomAttention()(td)
    att = Add()([td, att])
    att = LayerNormalization()(att)

    lstm_out = LSTM(64, return_sequences=False)(att)
    global_pool = GlobalAveragePooling1D()(td)

    concat = Concatenate()([lstm_out, global_pool])
    output = Dense(1, activation="sigmoid")(concat)

    model = Model(input_tensor, output)
    model.compile(
        optimizer=Adam(1e-5, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
