import numpy as np
from keras import layers, optimizers, models


def relu(x):
    if x > 0:
        return x
    return 0


def uNet(num_classes, input_shape):
    img_input = layers.Input(input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = layers.BatchNormalization()(x)
    block_1_out = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(block_1_out)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = layers.BatchNormalization()(x)
    block_2_out = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(block_2_out)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = layers.BatchNormalization()(x)
    block_3_out = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(block_3_out)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = layers.BatchNormalization()(x)
    block_4_out = layers.Activation('relu')(x)

    #x = layers.MaxPooling2D()(block_4_out)

    # Block 5
    #x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)


    #Load pretrained weights.
    #for_pretrained_weight = layers.MaxPooling2D()(x)
    #vgg16 = Model(img_input, for_pretrained_weight)
    #vgg16.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    # UP 1
    #x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    #x = np.concatenate([x, block_4_out])
    #x = layers.Conv2D(512, (3, 3), padding='same')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Conv2D(512, (3, 3), padding='same')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)

    # UP 2
    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name = 'layers.Conv2DTranspose_UP2')(block_4_out)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = np.concatenate([x, block_3_out])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # UP 3
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name = 'layers.Conv2DTranspose_UP3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = np.concatenate([x, block_2_out])
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # UP 4
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name = 'layers.Conv2DTranspose_UP4')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = np.concatenate([x, block_1_out])
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(img_input, x)
    model.compile(optimizer=optimizers.Adam(lr=0.005),
                  loss='categorical_crossentropy',
                  metrics=['dice_coef'])
    model.summary()
    return model
