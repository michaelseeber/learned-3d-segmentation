import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import segment from segmentation.py


EPSILON = 10e-8


def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)


def conv_weight_variable(name, shape, stddev=1.0):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def bias_weight_variable(name, shape, cval=0.0):
    initializer = tf.constant_initializer(cval)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def conv3d(x, weights, name=None):
    return tf.nn.conv3d(x, weights, name=name,
                        strides=[1, 1, 1, 1, 1], padding="SAME")


def conv3d_adj(x, weights, num_channels, name=None):
    output_shape = x.get_shape().as_list()
    output_shape[0] = tf.shape(x)[0]
    output_shape[4] = num_channels
    return tf.nn.conv3d_transpose(x, weights, name=name,
                                  output_shape=output_shape,
                                  strides=[1, 1, 1, 1, 1], padding="SAME")


def avg_pool3d(x, factor, name=None):
    return tf.nn.avg_pool3d(x,
                            ksize=[1, factor, factor, factor, 1],
                            strides=[1, factor, factor, factor, 1],
                            padding="SAME", name=name)


def max_pool3d(x, factor, name=None):
    return tf.nn.max_pool3d(x,
                            ksize=[1, factor, factor, factor, 1],
                            strides=[1, factor, factor, factor, 1],
                            padding="SAME", name=name)


def resize_volumes(x, row_factor, col_factor, slice_factor):
    output = repeat_elements(x, row_factor, axis=1)
    output = repeat_elements(output, col_factor, axis=2)
    output = repeat_elements(output, slice_factor, axis=3)
    return output


def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    return tf.concat([x for x in tensors], axis)


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.
    # Returns
        A tensor.
    """
    x_shape = x.get_shape().as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)

    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.

    # Repeating
    auxiliary_axis = axis + 1
    x_shape = tf.shape(x)
    x_rep = tf.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.get_shape()) + 1)
    reps[auxiliary_axis] = rep
    x_rep = tf.tile(x_rep, reps)

    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = tf.constant(reps, dtype='int32')
    x_shape = x_shape * reps
    x_rep = tf.reshape(x_rep, x_shape)

    # Fix shape representation
    x_shape = x.get_shape().as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)

    return x_rep


def update_lagrangian(u_, l, sig, level):
    assert len(u_) == len(l)

    with tf.name_scope("lagrange_update"):
        sum_u = tf.reduce_sum(u_[level], axis=4, keep_dims=False)
        l[level] += sig * (sum_u - 1.0)


def update_dual(u_, m, sig, level):
    assert len(u_) == len(m)

    with tf.name_scope("dual_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        _, nrows, ncols, nslices, nclasses = \
            u_[level].get_shape().as_list()
        batch_size = tf.shape(u_[level])[0]

        if level + 1 < len(u_):
            grad_u1 = conv3d(u_[level], w1)
            grad_u2 = conv3d(u_[level + 1], w2)
            grad_u = grad_u1 + resize_volumes(grad_u2, 2, 2, 2)
        else:
            grad_u = conv3d(u_[level], w1)

        m[level] += sig * grad_u

        m_rshp = tf.reshape(m[level], [batch_size, nrows, ncols,
                                       nslices, nclasses, 3])

        m_norm = tf.norm(m_rshp, ord="euclidean", axis=5, keep_dims=True)
        m_norm = tf.maximum(m_norm, 1.0)
        m_normalized = tf.divide(m_rshp, m_norm)

        m[level] = tf.reshape(m_normalized, [batch_size, nrows, ncols,
                                             nslices, nclasses * 3])


def update_primal(u, m, l, d, tau, level):
    assert len(u) == len(m)
    assert len(u) == len(l)
    assert len(u) == len(d)

    with tf.name_scope("primal_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        # u_shape = tf.shape(u[level])
        # batch_size = u_shape[0]
        # nrows = u_shape[1]
        # ncols = u_shape[2]
        # nslices = u_shape[3]
        # nclasses = u_shape[4]

        _, nrows, ncols, nslices, nclasses = \
            u[level].get_shape().as_list()
        batch_size = tf.shape(u[level])[0]

        if level + 1 < len(u):
            div_m1 = conv3d_adj(m[level], w1, nclasses)
            div_m2 = conv3d_adj(m[level + 1], w2, nclasses)
            div_m = div_m1 + resize_volumes(div_m2, 2, 2, 2)
        else:
            div_m = conv3d_adj(m[level], w1, nclasses)

        l_rshp = tf.reshape(l[level], [batch_size, nrows, ncols, nslices, 1])

        u[level] -= tau * (d[level] + l_rshp + div_m)

        u[level] = tf.minimum(1.0, tf.maximum(u[level], 0.0))


def primal_dual(u, u_, m, l, d, sig, tau, iter):
    u_0 = list(u)

    for level in list(range(len(u)))[::-1]:
        with tf.name_scope("primal_dual_iter{}_level{}".format(iter, level)):
            update_dual(u_, m, sig, level)

            update_lagrangian(u_, l, sig, level)

            update_primal(u, m, l, d, tau, level)

            u_[level] = 2 * u[level] - u_0[level]

    return u, u_, m, l


def classification_accuracy(y_true, y_pred):
    with tf.name_scope("classification_accuracy"):
        labels_true = tf.argmax(y_true, axis=-1)
        labels_pred = tf.argmax(y_pred, axis=-1)

        freespace_label = y_true.shape[-1] - 1
        unkown_label = y_true.shape[-1] - 2

        freespace_mask = tf.equal(labels_true, freespace_label)
        unknown_mask = tf.equal(labels_true, unkown_label)
        not_unobserved_mask = tf.reduce_any(tf.greater(y_true, 0.5), axis=-1)
        occupied_mask = ~freespace_mask & ~unknown_mask & not_unobserved_mask

        freespace_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(labels_true, freespace_mask),
            tf.boolean_mask(labels_pred, freespace_mask)), dtype=tf.float32))
        occupied_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(tf.less(labels_true, freespace_label),
                            occupied_mask),
            tf.boolean_mask(tf.less(labels_pred, freespace_label),
                            occupied_mask)), dtype=tf.float32))
        semantic_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(labels_true, occupied_mask),
            tf.boolean_mask(labels_pred, occupied_mask)), dtype=tf.float32))

    return freespace_accuracy, occupied_accuracy, semantic_accuracy


def categorical_crossentropy(y_true, y_pred, params):
    nclasses = y_true.shape[-1]

    with tf.name_scope("categorical_cross_entropy"):
        y_true = tf.nn.softmax(params["softmax_scale"] * y_true)
        # y_pred = tf.nn.softmax(params["softmax_scale"] * y_pred)

        y_true = tf.clip_by_value(y_true, EPSILON, 1.0 - EPSILON)
        y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)

        # Measure how close we are to unknown class.
        unkown_weights = 1 - y_true[..., -2][..., None]

        # Measure how close we are to unobserved class by measuring KL
        # divergence to uniform distribution.
        unobserved_weights = tf.log(tf.cast(nclasses, tf.float32)) + \
                             tf.reduce_sum(y_true * tf.log(y_true),
                                           axis=-1, keep_dims=True)

        # Final per voxel loss function weights.
        weights = tf.maximum(EPSILON, unkown_weights * unobserved_weights)

        # Compute weighted cross entropy.
        cross_entropy = -tf.reduce_sum(weights * y_true * tf.log(y_pred)) / \
                         tf.reduce_sum(weights)

    return cross_entropy


def build_model(params):
    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    softmax_scale = params["softmax_scale"]

    # Setup placeholders and variables.

    d = tf.placeholder(tf.float32, [None, nrows, ncols,
                                    nslices, nclasses], name="d")

    u = []
    u_ = []
    m = []
    l = []
    for level in range(nlevels):
        factor = 2 ** level
        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0
        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor
        u.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, nclasses],
                     name="u{}".format(level)))
        u_.append(tf.placeholder(
                      tf.float32, [None, nrows_level, ncols_level,
                                   nslices_level, nclasses],
                     name="u_{}".format(level)))
        m.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, 3 * nclasses],
                     name="m{}".format(level)))
        l.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level],
                     name="l{}".format(level)))

        with tf.variable_scope("weights{}".format(level)):
            conv_weight_variable(
                "w1", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)
            conv_weight_variable(
                "w2", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)

    sig = params["sig"]
    tau = params["tau"]
    lam = params["lam"]
    niter = params["niter"]

    d_lam = tf.multiply(d, lam, name="d_lam")

    d_encoded = []
    for level in range(nlevels):
        with tf.name_scope("datacost_encoding{}".format(level)):
            if level > 0:
                d_lam = avg_pool3d(d_encoded[level - 1], 2)

            with tf.variable_scope("weights{}".format(level)):
                w1_d = conv_weight_variable(
                    "w1_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w2_d = conv_weight_variable(
                    "w2_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w3_d = conv_weight_variable(
                    "w3_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                b1_d = bias_weight_variable("b1_d", [nclasses])
                b2_d = bias_weight_variable("b2_d", [nclasses])
                b3_d = bias_weight_variable("b3_d", [nclasses])

            d_residual = conv3d(d_lam, w1_d)
            d_residual = tf.nn.relu(d_residual + b1_d)
            d_residual = conv3d(d_residual, w2_d)
            d_residual = tf.nn.relu(d_residual + b2_d)
            d_residual = conv3d(d_residual, w3_d,
                                name="d_encoded{}".format(level))
            d_residual += b3_d

            d_encoded.append(d_lam + d_residual)

    # Create a copy of the placeholders for the loop variables.
    u_loop = list(u)
    u_loop_= list(u_)
    m_loop = list(m)
    l_loop = list(l)

    for iter in range(niter):
        u_loop, u_loop_, m_loop, l_loop = primal_dual(
            u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)

    probs = u_loop

    for level in range(nlevels):
        with tf.name_scope("prob_decoding{}".format(level)):
            with tf.variable_scope("weights{}".format(level)):
                w1_p = conv_weight_variable(
                    "w1_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w2_p = conv_weight_variable(
                    "w2_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w3_p = conv_weight_variable(
                    "w3_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                b1_p = bias_weight_variable("b1_p", [nclasses])
                b2_p = bias_weight_variable("b2_p", [nclasses])
                b3_p = bias_weight_variable("b3_p", [nclasses])

            probs_residual = conv3d(probs[level], w1_p)
            probs_residual = tf.nn.relu(probs_residual + b1_p)
            probs_residual = conv3d(probs_residual, w2_p)
            probs_residual = tf.nn.relu(probs_residual + b2_p)
            probs_residual = conv3d(probs_residual, w3_p)
            probs_residual += b3_p

            probs[level] = tf.nn.softmax(softmax_scale * probs[level]
                                         + probs_residual,
                                         name="probs{}".format(level))

    return probs, d, u, u_, m, l


def build_data_generator(scene_path, scene_list_path, params):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]

    scene_list = []
    with open(scene_list_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)

    # Load the data for all scenes.
    datacosts = []
    groundtruths = []
    for i, scene_name in enumerate(scene_list):
        print("Loading {} [{}/{}]".format(scene_name, i + 1, len(scene_list)))

        # if len(datacosts) == 5:
        #     break

        datacost_path = os.path.join(scene_path, scene_name,"converted",
                                     "datacost.npz")
        groundtruth_path = os.path.join(scene_path, scene_name, "converted",
                                        "groundtruth_model/probs.npz")

        if not os.path.exists(datacost_path):
            print("  Warning: datacost does not exist: {}".format(datacost_path))
            continue

        if not os.path.exists(groundtruth_path):
            print("  Warning: groundtruth does not exist: {}".format(groundtruth_path))
            continue

        datacost_data = np.load(datacost_path)
        datacost = datacost_data["volume"]

        groundtruth = np.load(groundtruth_path)["probs"]

        # Make sure the data is compatible with the parameters.
        print(datacost_path)
        print(datacost.shape)
        assert datacost.shape[3] == nclasses
        assert datacost.shape == groundtruth.shape

        datacosts.append(datacost)
        groundtruths.append(groundtruth)

    idxs = np.arange(len(datacosts))

    batch_datacost = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)
    batch_groundtruth = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)

    npasses = 0

    while True:
        # Shuffle all data samples.
        np.random.shuffle(idxs)

        # One epoch iterates once over all scenes.
        for batch_start_idx in range(0, len(idxs), batch_size):
            # Determine the random scenes for current batch.
            batch_end_idx = min(batch_start_idx + batch_size, len(idxs))
            batch_idxs = idxs[batch_start_idx:batch_end_idx]

            # By default, set all voxels to unobserved.
            batch_datacost[:] = 0
            batch_groundtruth[:] = 1.0 / nclasses

            # Prepare data for random scenes in current batch.
            for i, idx in enumerate(batch_idxs):
                datacost = datacosts[idx]
                groundtruth = groundtruths[idx]

                # Determine a random crop of the input data.
                row_start = np.random.randint(
                    0, max(datacost.shape[0] - nrows, 0) + 1)
                col_start = np.random.randint(
                    0, max(datacost.shape[1] - ncols, 0) + 1)
                slice_start = np.random.randint(
                    0, max(datacost.shape[2] - nslices, 0) + 1)
                row_end = min(row_start + nrows, datacost.shape[0])
                col_end = min(col_start + ncols, datacost.shape[1])
                slice_end = min(slice_start + nslices, datacost.shape[2])

                # Copy the random crop of the data cost.
                batch_datacost[i,
                               :row_end-row_start,
                               :col_end-col_start,
                               :slice_end-slice_start] = \
                    datacost[row_start:row_end,
                             col_start:col_end,
                             slice_start:slice_end]

                # Copy the random crop of the groundtruth.
                batch_groundtruth[i,
                                  :row_end-row_start,
                                  :col_end-col_start,
                                  :slice_end-slice_start] = \
                    groundtruth[row_start:row_end,
                                col_start:col_end,
                                slice_start:slice_end]

                # Randomly rotate around z-axis.
                num_rot90 = np.random.randint(4)
                if num_rot90 > 0:
                    batch_datacost[i] = np.rot90(batch_datacost[i],
                                                 k=num_rot90,
                                                 axes=(0, 1))
                    batch_groundtruth[i] = np.rot90(batch_groundtruth[i],
                                                    k=num_rot90,
                                                    axes=(0, 1))

                # Randomly flip along x and y axis.
                flip_axis = np.random.randint(3)
                if flip_axis == 0 or flip_axis == 1:
                    batch_datacost[i] = np.flip(batch_datacost[i],
                                                axis=flip_axis)
                    batch_groundtruth[i] = np.flip(batch_groundtruth[i],
                                                   axis=flip_axis)

            yield (batch_datacost[:len(batch_idxs)],
                   batch_groundtruth[:len(batch_idxs)])

        npasses += 1

        if epoch_npasses > 0 and npasses >= epoch_npasses:
            npasses = 0
            yield

def build_data_generator_pointNet(scene_path, scene_list, params):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]

    scene_list = []
    with open(scene_list_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)

    # Load the data for all scenes.
    datacosts = []
    groundtruths = []
    for i, scene_name in enumerate(scene_list):
        print("Loading {} [{}/{}]".format(scene_name, i + 1, len(scene_list)))

        # if len(datacosts) == 5:
        #     break

        datacost_path = os.path.join(scene_path, scene_name,"converted",
                                     "datacost.npz")
        groundtruth_path = os.path.join(scene_path, scene_name, "converted",
                                        "groundtruth_model/probs.npz")

        if not os.path.exists(datacost_path):
            print("  Warning: datacost does not exist: {}".format(datacost_path))
            continue

        if not os.path.exists(groundtruth_path):
            print("  Warning: groundtruth does not exist: {}".format(groundtruth_path))
            continue

        datacost_data = np.load(datacost_path)
        datacost = datacost_data["volume"]

        groundtruth = np.load(groundtruth_path)["probs"]

        # Make sure the data is compatible with the parameters.
        print(datacost_path)
        print(datacost.shape)
        assert datacost.shape[3] == nclasses
        assert datacost.shape == groundtruth.shape

        datacosts.append(datacost)
        groundtruths.append(groundtruth)

    idxs = np.arange(len(datacosts))

    batch_datacost = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)
    batch_groundtruth = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)

    npasses = 0

    while True:
        # Shuffle all data samples.
        np.random.shuffle(idxs)

        # One epoch iterates once over all scenes.
        for batch_start_idx in range(0, len(idxs), batch_size):
            # Determine the random scenes for current batch.
            batch_end_idx = min(batch_start_idx + batch_size, len(idxs))
            batch_idxs = idxs[batch_start_idx:batch_end_idx]

            # By default, set all voxels to unobserved.
            batch_datacost[:] = 0
            batch_groundtruth[:] = 1.0 / nclasses

            # Prepare data for random scenes in current batch.
            for i, idx in enumerate(batch_idxs):
                datacost = datacosts[idx]
                groundtruth = groundtruths[idx]

                # Determine a random crop of the input data.
                row_start = np.random.randint(
                    0, max(datacost.shape[0] - nrows, 0) + 1)
                col_start = np.random.randint(
                    0, max(datacost.shape[1] - ncols, 0) + 1)
                slice_start = np.random.randint(
                    0, max(datacost.shape[2] - nslices, 0) + 1)
                row_end = min(row_start + nrows, datacost.shape[0])
                col_end = min(col_start + ncols, datacost.shape[1])
                slice_end = min(slice_start + nslices, datacost.shape[2])

                # Copy the random crop of the data cost.
                batch_datacost[i,
                               :row_end-row_start,
                               :col_end-col_start,
                               :slice_end-slice_start] = \
                    datacost[row_start:row_end,
                             col_start:col_end,
                             slice_start:slice_end]

                # Copy the random crop of the groundtruth.
                batch_groundtruth[i,
                                  :row_end-row_start,
                                  :col_end-col_start,
                                  :slice_end-slice_start] = \
                    groundtruth[row_start:row_end,
                                col_start:col_end,
                                slice_start:slice_end]

                # Randomly rotate around z-axis.
                num_rot90 = np.random.randint(4)
                if num_rot90 > 0:
                    batch_datacost[i] = np.rot90(batch_datacost[i],
                                                 k=num_rot90,
                                                 axes=(0, 1))
                    batch_groundtruth[i] = np.rot90(batch_groundtruth[i],
                                                    k=num_rot90,
                                                    axes=(0, 1))

                # Randomly flip along x and y axis.
                flip_axis = np.random.randint(3)
                if flip_axis == 0 or flip_axis == 1:
                    batch_datacost[i] = np.flip(batch_datacost[i],
                                                axis=flip_axis)
                    batch_groundtruth[i] = np.flip(batch_groundtruth[i],
                                                   axis=flip_axis)

            yield (batch_datacost[:len(batch_idxs)],
                   batch_groundtruth[:len(batch_idxs)])

        npasses += 1

        if epoch_npasses > 0 and npasses >= epoch_npasses:
            npasses = 0
            yield

def train_model(scene_path, scene_train_list_path, scene_val_list_path,
                model_path, params):
    log_path = os.path.join(model_path, "logs")
    checkpoint_path = os.path.join(model_path, "checkpoint")

    mkdir_if_not_exists(model_path)
    mkdir_if_not_exists(log_path)
    mkdir_if_not_exists(checkpoint_path)

    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    val_nbatches = params["val_nbatches"]

    train_data_generator = \
        build_data_generator(scene_path, scene_train_list_path, params)

    val_params = dict(params)
    val_params["epoch_npasses"] = -1
    val_data_generator = \
        build_data_generator(scene_path, scene_val_list_path, val_params)

    probs, datacost, u, u_, m, l = build_model(params)
    groundtruth = tf.placeholder(tf.float32, probs[0].shape, name="groundtruth")

    u_init = []
    u_init_ = []
    m_init = []
    l_init = []
    for level in range(nlevels):
        factor = 2 ** level
        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0
        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor
        u_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level, nclasses],
                               dtype=np.float32))
        u_init_.append(np.empty([batch_size, nrows_level, ncols_level,
                                 nslices_level, nclasses],
                                dtype=np.float32))
        m_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level, 3 * nclasses],
                               dtype=np.float32))
        l_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level],
                               dtype=np.float32))

    loss_op = categorical_crossentropy(groundtruth, probs[0], params)
    freespace_accuracy_op, occupied_accuracy_op, semantic_accuracy_op = \
        classification_accuracy(groundtruth, probs[0])

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss_op)

    train_loss_summary = \
        tf.placeholder(tf.float32, name="train_loss_summary")
    train_freespace_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_freespace_accuracy_summary")
    train_occupied_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_occupied_accuracy_summary")
    train_semantic_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_semantic_accuracy_summary")

    tf.summary.scalar("train_loss",
                      train_loss_summary)
    tf.summary.scalar("train_freespace_accuracy",
                      train_freespace_accuracy_summary)
    tf.summary.scalar("train_occupied_accuracy",
                      train_occupied_accuracy_summary)
    tf.summary.scalar("train_semantic_accuracy",
                      train_semantic_accuracy_summary)

    val_loss_summary = \
        tf.placeholder(tf.float32, name="val_loss_summary")
    val_freespace_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_freespace_accuracy_summary")
    val_occupied_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_occupied_accuracy_summary")
    val_semantic_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_semantic_accuracy_summary")

    tf.summary.scalar("val_loss",
                      val_loss_summary)
    tf.summary.scalar("val_freespace_accuracy",
                      val_freespace_accuracy_summary)
    tf.summary.scalar("val_occupied_accuracy",
                      val_occupied_accuracy_summary)
    tf.summary.scalar("val_semantic_accuracy",
                      val_semantic_accuracy_summary)

    summary_op = tf.summary.merge_all()

    model_saver = tf.train.Saver(save_relative_paths=True)
    train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2,
                                 save_relative_paths=True, pad_step_number=True)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        model_saver.save(sess, os.path.join(checkpoint_path, "initial"),
                         write_meta_graph=True)

        for epoch in range(params["nepochs"]):

            train_loss_values = []
            train_freespace_accuracy_values = []
            train_occupied_accuracy_values = []
            train_semantic_accuracy_values = []

            batch = 0
            while True:
                train_data = next(train_data_generator)

                # Check if epoch finished.
                if train_data is None:
                    break

                datacost_batch, groundtruth_batch = train_data

                num_batch_samples = datacost_batch.shape[0]

                feed_dict = {}

                feed_dict[datacost] = datacost_batch
                feed_dict[groundtruth] = groundtruth_batch

                for level in range(nlevels):
                    u_init[level][:] = 1.0 / nclasses
                    u_init_[level][:] = 1.0 / nclasses
                    m_init[level][:] = 0.0
                    l_init[level][:] = 0.0
                    feed_dict[u[level]] = u_init[level][:num_batch_samples]
                    feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
                    feed_dict[m[level]] = m_init[level][:num_batch_samples]
                    feed_dict[l[level]] = l_init[level][:num_batch_samples]

                (_,
                 loss,
                 freespace_accuracy,
                 occupied_accuracy,
                 semantic_accuracy) = sess.run(
                    [train_op,
                     loss_op,
                     freespace_accuracy_op,
                     occupied_accuracy_op,
                     semantic_accuracy_op],
                    feed_dict=feed_dict
                )

                train_loss_values.append(loss)
                train_freespace_accuracy_values.append(freespace_accuracy)
                train_occupied_accuracy_values.append(occupied_accuracy)
                train_semantic_accuracy_values.append(semantic_accuracy)

                print("Epoch: {}, "
                      "Batch: {}\n"
                      "  Loss:                  {}\n"
                      "  Free Space Accuracy:   {}\n"
                      "  Occupied Accuracy:     {}\n"
                      "  Semantic Accuracy:     {}".format(
                      epoch + 1,
                      batch + 1,
                      loss,
                      freespace_accuracy,
                      occupied_accuracy,
                      semantic_accuracy))

                batch += 1

            train_loss_value = \
                np.nanmean(train_loss_values)
            train_freespace_accuracy_value = \
                np.nanmean(train_freespace_accuracy_values)
            train_occupied_accuracy_value = \
                np.nanmean(train_occupied_accuracy_values)
            train_semantic_accuracy_value = \
                np.nanmean(train_semantic_accuracy_values)

            val_loss_values = []
            val_freespace_accuracy_values = []
            val_occupied_accuracy_values = []
            val_semantic_accuracy_values = []

            for _ in range(val_nbatches):
                datacost_batch, groundtruth_batch = next(val_data_generator)

                num_batch_samples = datacost_batch.shape[0]

                feed_dict = {}

                feed_dict[datacost] = datacost_batch
                feed_dict[groundtruth] = groundtruth_batch

                for level in range(nlevels):
                    u_init[level][:] = 1.0 / nclasses
                    u_init_[level][:] = 1.0 / nclasses
                    m_init[level][:] = 0.0
                    l_init[level][:] = 0.0
                    feed_dict[u[level]] = u_init[level][:num_batch_samples]
                    feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
                    feed_dict[m[level]] = m_init[level][:num_batch_samples]
                    feed_dict[l[level]] = l_init[level][:num_batch_samples]

                (loss,
                 freespace_accuracy,
                 occupied_accuracy,
                 semantic_accuracy) = sess.run(
                    [loss_op,
                     freespace_accuracy_op,
                     occupied_accuracy_op,
                     semantic_accuracy_op],
                    feed_dict=feed_dict
                )

                val_loss_values.append(loss)
                val_freespace_accuracy_values.append(freespace_accuracy)
                val_occupied_accuracy_values.append(occupied_accuracy)
                val_semantic_accuracy_values.append(semantic_accuracy)

            val_loss_value = \
                np.nanmean(val_loss_values)
            val_freespace_accuracy_value = \
                np.nanmean(val_freespace_accuracy_values)
            val_occupied_accuracy_value = \
                np.nanmean(val_occupied_accuracy_values)
            val_semantic_accuracy_value = \
                np.nanmean(val_semantic_accuracy_values)

            print()
            print(78 * "#")
            print()

            print("Validation\n"
                  "  Loss:                  {}\n"
                  "  Free Space Accuracy:   {}\n"
                  "  Occupied Accuracy:     {}\n"
                  "  Semantic Accuracy:     {}".format(
                  val_loss_value,
                  val_freespace_accuracy_value,
                  val_occupied_accuracy_value,
                  val_semantic_accuracy_value))

            print()
            print(78 * "#")
            print(78 * "#")
            print()

            summary = sess.run(
                summary_op,
                feed_dict={
                    train_loss_summary:
                        train_loss_value,
                    train_freespace_accuracy_summary:
                        train_freespace_accuracy_value,
                    train_occupied_accuracy_summary:
                        train_occupied_accuracy_value,
                    train_semantic_accuracy_summary:
                        train_semantic_accuracy_value,
                    val_loss_summary:
                        val_loss_value,
                    val_freespace_accuracy_summary:
                        val_freespace_accuracy_value,
                    val_occupied_accuracy_summary:
                        val_occupied_accuracy_value,
                    val_semantic_accuracy_summary:
                        val_semantic_accuracy_value,
                }
            )

            log_writer.add_summary(summary, epoch)

            train_saver.save(sess, os.path.join(checkpoint_path, "checkpoint"),
                             global_step=epoch, write_meta_graph=False)

        model_saver.save(sess, os.path.join(checkpoint_path, "final"),
                         write_meta_graph=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--scene_train_list_path", required=True)
    parser.add_argument("--scene_val_list_path", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--nclasses", type=int, required=True)

    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--nrows", type=int, default=24)
    parser.add_argument("--ncols", type=int, default=24)
    parser.add_argument("--nslices", type=int, default=24)

    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--epoch_npasses", type=int, default=1)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--softmax_scale", type=float, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "nepochs": args.nepochs,
        "epoch_npasses": args.epoch_npasses,
        "val_nbatches": args.val_nbatches,
        "batch_size": args.batch_size,
        "nlevels": args.nlevels,
        "nclasses": args.nclasses,
        "nrows": args.nrows,
        "ncols": args.ncols,
        "nslices": args.nslices,
        "niter": args.niter,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "softmax_scale": args.softmax_scale,
    }

    train_model(args.scene_path, args.scene_train_list_path,
                args.scene_val_list_path, args.model_path, params)


if __name__ == "__main__":
    main()
