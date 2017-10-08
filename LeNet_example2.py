import tensorflow as tf


def LeNet(x, mu=0, sigma=0.1,
          conv1_filter=5, conv1_depth=6, conv1_strides=1, pool1_ksize=2,
          pool1_strides=2, conv2_filter=5, conv2_depth=16, conv2_strides=1,
          pool2_ksize=2, pool2_strides=2, layer3_depth=120, layer4_depth=84):
    """
    This example presents variables for shape to help understand calculations
    """
    n_classes = 10

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    s = (conv1_filter, conv1_filter, 3, conv1_depth)
    conv1_W = tf.Variable(tf.truncated_normal(shape=s, mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_depth))
    std = [1, conv1_strides, conv1_strides, 1]
    conv1 = tf.nn.conv2d(x, conv1_W, strides=std, padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    ks = [1, pool1_ksize, pool1_ksize, 1]
    std = [1, pool1_strides, pool1_strides, 1]
    conv1 = tf.nn.max_pool(conv1, ksize=ks, strides=std, padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    s = (conv2_filter, conv2_filter, conv1_depth, conv2_depth)
    conv2_W = tf.Variable(tf.truncated_normal(shape=s, mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(conv2_depth))
    std = [1, conv2_strides, conv2_strides, 1]
    p = 'VALID'
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=std, padding=p) + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    ks = [1, pool2_ksize, pool2_ksize, 1]
    std = [1, pool2_strides, pool2_strides, 1]
    conv2 = tf.nn.max_pool(conv2, ksize=ks, strides=std, padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    s = (400, layer3_depth)
    fc1_W = tf.Variable(tf.truncated_normal(shape=s, mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(layer3_depth))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    s = (layer3_depth, layer4_depth)
    fc2_W = tf.Variable(tf.truncated_normal(shape=s, mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(layer4_depth))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    s = (layer4_depth, n_classes)
    fc3_W = tf.Variable(tf.truncated_normal(shape=s, mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
