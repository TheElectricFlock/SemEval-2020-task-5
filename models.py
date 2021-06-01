import numpy as np
from tensorflow.keras import datasets, layers, models, losses, callbacks, Model, backend


def get_SegNet(is_deep=False, is_sko=False):
    
    if is_sko:
        final_output_layer = 1
    else:
        final_output_layer = 9
        
    model = models.Sequential()
    model.add(layers.Input(shape = train_X.shape[1:]))

    model.add(layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))

    model.add(layers.Conv1D(filters=64, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))
    model.add(layers.Dropout(0.3))
    
    if is_deep:
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv1D(filters=512, kernel_size=9, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ELU())
        model.add(layers.MaxPooling1D(strides=2))
        model.add(layers.Dropout(0.3))

        model.add(layers.UpSampling1D(size=2))
        model.add(layers.Conv1D(filters=512, kernel_size=9, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ELU())
        model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv1D(filters=3, kernel_size=final_output_layer, strides=1, padding='same', activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def get_DeconvNet(is_deep=False, is_sko=False):
    
    if is_sko:
        final_output_layer = 1
    else:
        final_output_layer = 9
        
    model = models.Sequential()
    model.add(layers.Input(shape = train_X.shape[1:]))

    model.add(layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))

    model.add(layers.Conv1D(filters=64, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.MaxPooling1D(strides=2))
    model.add(layers.Dropout(0.3))
    
    if is_deep:
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv1D(filters=512, kernel_size=9, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ELU())
        model.add(layers.MaxPooling1D(strides=2))
        model.add(layers.Dropout(0.3))

        model.add(layers.UpSampling1D(size=2))
        model.add(layers.Conv1DTranspose(filters=512, kernel_size=9, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ELU())
        model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(filters=256, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(filters=128, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(filters=64, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(filters=32, kernel_size=9, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv1D(filters=3, kernel_size=final_output_layer, strides=1, padding='same', activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def get_UNet(is_deep=False, is_sko=False):
    
    if is_sko:
        final_output_layer = 1
    else:
        final_output_layer = 9
        
    if is_deep:
        encoder = [64, 128, 256, 512]
        decoder = [512, 256, 128, 64, 32]
    else:
        encoder = [64, 128, 256]
        decoder = [256, 128, 64, 32]
        
    # Entry conv block
    inputs = layers.Input(shape = train_X.shape[1:])
    x = layers.Conv1D(32, 9, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling1D(strides=2, padding="same")(x)
    
    previous_block_activation = x
    
    for filters in encoder:
        x = layers.Conv1D(filters, 9, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ELU()(x)
        x = layers.MaxPooling1D(strides=2, padding="same")(x)

        residual = layers.Conv1D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    for filters in decoder:
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1DTranspose(filters, 9, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ELU()(x)

        # Project residual
        residual = layers.UpSampling1D(size=2)(previous_block_activation)
        residual = layers.Conv1D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Classification layer
    outputs = layers.Conv1D(3, final_output_layer, activation="softmax", padding="same")(x)
    
    # Define the model
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model