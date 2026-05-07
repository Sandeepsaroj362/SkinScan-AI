import base64
import io

import numpy as np
import tensorflow as tf

from PIL import Image


def find_last_conv_layer_name(model):

    for layer in reversed(model.layers):

        if hasattr(layer, "layers") and layer.layers:

            try:
                return find_last_conv_layer_name(layer)

            except ValueError:
                pass

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            return layer.name

    raise ValueError(
        "No convolutional layer found in the model."
    )

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None,
):

    base_model = model.get_layer(
        "efficientnetb0"
    )

    classifier_layer_names = [
        "global_average_pooling2d",
        "dropout",
        "dense",
    ]

    last_conv_layer = (
        base_model.get_layer(
            last_conv_layer_name
        )
    )

    last_conv_layer_model = tf.keras.Model(
        base_model.input,
        last_conv_layer.output,
    )

    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    for layer_name in classifier_layer_names:

        x = model.get_layer(
            layer_name
        )(x)

    classifier_model = tf.keras.Model(
        classifier_input,
        x,
    )

    with tf.GradientTape() as tape:

        last_conv_layer_output = (
            last_conv_layer_model(
                img_array
            )
        )

        tape.watch(
            last_conv_layer_output
        )

        preds = classifier_model(
            last_conv_layer_output
        )

        if pred_index is None:

            pred_index = tf.argmax(
                preds[0]
            )

        class_channel = preds[
            :, pred_index
        ]

    grads = tape.gradient(
        class_channel,
        last_conv_layer_output,
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    last_conv_layer_output = (
        last_conv_layer_output[0]
    )

    heatmap = (
        last_conv_layer_output
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0,
    ) / (
        tf.math.reduce_max(
            heatmap
        )
        + 1e-8
    )

    return heatmap.numpy()


def create_gradcam_image(
    img_path,
    model,
    target_size=(224, 224),
    alpha=0.45,
):

    img = tf.keras.utils.load_img(
        img_path,
        target_size=target_size,
    )

    img_array = (
        tf.keras.utils.img_to_array(img)
    )

    img_array = np.expand_dims(
        img_array,
        axis=0,
    )

    # EfficientNet base model
    base_model = model.get_layer(
        "efficientnetb0"
    )

    # Find last conv layer INSIDE EfficientNet
    last_conv_layer_name = (
        find_last_conv_layer_name(
            base_model
        )
    )

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
    )

    original_image = (
        Image.open(img_path)
        .convert("RGB")
        .resize(target_size)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_img = (
        Image.fromarray(heatmap)
        .resize(target_size)
    )

    heatmap = np.array(
        heatmap_img
    )

    colored = np.zeros(
        (
            target_size[1],
            target_size[0],
            3,
        ),
        dtype=np.uint8,
    )

    # Red heatmap overlay
    colored[..., 0] = heatmap

    colored[..., 1] = (
        heatmap * 0.25
    ).astype(np.uint8)

    colored[..., 2] = (
        255 - heatmap
    )

    overlay = Image.blend(
        original_image,
        Image.fromarray(colored),
        alpha=alpha,
    )

    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        f"data:image/png;base64,{encoded}"
    )