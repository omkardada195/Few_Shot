import numpy as np
import tensorflow as tf

from src.model_utils import build_small_unet, bce_dice_loss, dice_coef


def clone_model_with_weights(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.build(model.input_shape)
    new_model.set_weights(model.get_weights())
    return new_model


def create_reptile_base_model(input_shape=(192, 192, 1)):
    model = build_small_unet(input_shape=input_shape)
    return model


def inner_train_step(model, X, Y, inner_lr=1e-3, inner_epochs=1, batch_size=2):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=inner_lr),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    model.fit(
        X, Y,
        epochs=inner_epochs,
        batch_size=min(batch_size, len(X)),
        verbose=0
    )
    return model


def reptile_meta_update(meta_model, task_model, outer_lr=0.1):
    meta_weights = meta_model.get_weights()
    task_weights = task_model.get_weights()

    new_weights = []
    for w_meta, w_task in zip(meta_weights, task_weights):
        new_w = w_meta + outer_lr * (w_task - w_meta)
        new_weights.append(new_w)

    meta_model.set_weights(new_weights)
    return meta_model


def adapt_reptile_model(
    meta_model,
    X_support,
    Y_support,
    inner_lr=1e-3,
    inner_epochs=3,
    batch_size=2
):
    adapted = clone_model_with_weights(meta_model)
    adapted = inner_train_step(
        adapted,
        X_support,
        Y_support,
        inner_lr=inner_lr,
        inner_epochs=inner_epochs,
        batch_size=batch_size
    )
    return adapted