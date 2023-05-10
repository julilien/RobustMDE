import tensorflow as tf


def flip_coin():
    return tf.random.uniform([]) > 0.5


def color_augmentation(image, depth):
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image, depth


def swap_color_channels(image, depth):
    image = tf.cond(flip_coin(), lambda: image[..., ::-1], lambda: image)

    return image, depth


def flip_image_horizontally(image, depth):
    do_flip = flip_coin()

    image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    depth = tf.cond(do_flip, lambda: tf.image.flip_left_right(tf.expand_dims(tf.squeeze(depth), axis=-1)),
                    lambda: depth)

    return image, tf.squeeze(depth)
