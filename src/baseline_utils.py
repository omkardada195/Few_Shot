import numpy as np
import tensorflow as tf

from src.model_utils import build_small_unet, bce_dice_loss, dice_coef


def create_baseline_model(input_shape=(192, 192, 1), learning_rate=1e-3):
    model = build_small_unet(input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    return model


def train_baseline_on_support(
    X_support,
    Y_support,
    epochs=8,
    batch_size=2,
    learning_rate=1e-3,
    verbose=0
):
    model = create_baseline_model(
        input_shape=X_support.shape[1:],
        learning_rate=learning_rate
    )

    history = model.fit(
        X_support,
        Y_support,
        epochs=epochs,
        batch_size=min(batch_size, len(X_support)),
        verbose=verbose
    )
    return model, history


def evaluate_model(model, X_query, Y_query, batch_size=4):
    results = model.evaluate(X_query, Y_query, batch_size=batch_size, verbose=0)
    names = model.metrics_names
    return {name: float(val) for name, val in zip(names, results)}


def predict_binary_masks(model, X, threshold=0.5):
    preds = model.predict(X, verbose=0)
    return (preds > threshold).astype(np.float32), preds