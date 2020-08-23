from keras import layers, models, optimizers


def build_model(input_shape, first_filters, kernel_size=(3, 3), blocks=3, middle_size=2):
    layer = layers.Input(input_shape)
    input_layer = layer
    max_filters = first_filters * 2 ** (blocks-1)

    # Convolution blocks
    convs = []
    for block in range(blocks):
        filters = first_filters * 2 ** block
        dropout = 0.25 * (block+1)
        if dropout > 0.5:
            dropout = 0.5

        layer = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(layer)
        layer = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(layer)
        convs.append(layer)
        layer = layers.MaxPooling2D((2, 2))(layer)
        layer = layers.Dropout(dropout)(layer)

    # Middle block
    for n in range(middle_size):
        layer = layers.Conv2D(first_filters * (2 ** blocks), kernel_size, activation='relu', padding='same')(layer)

    # Deconvolution blocks
    dropout = 0.5
    for block in range(blocks):
        filters = max_filters // 2 ** block

        layer = layers.Conv2DTranspose(filters, kernel_size, strides=(kernel_size[0] - 1, kernel_size[1] - 1), padding='same')(layer)
        layer = layers.concatenate([layer, convs[-block+1]])
        layer = layers.Dropout(dropout)(layer)
        layer = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(layer)
        layer = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(layer)

    # Output layer
    output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')(layer)
    model = models.Model(input_layer, output_layer)
    model.summary()
    return model


