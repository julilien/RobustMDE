import tensorflow as tf
import mlflow

from robustdepth.data.dao.diode import DIODETFDataAccessObject
from robustdepth.data.dao.ibims import IbimsTFDataAccessObject
from robustdepth.data.dao.nyu import NYULargeScaleTFDataAccessObject
from robustdepth.data.io_utils import Dataset
from robustdepth.losses.losses_meta import is_si_error
from robustdepth.metrics.depth_metrics import RootMeanSquaredError, DistanceUncensoredError, CroppingError, \
    ResizedError, DistanceCensoredError, AbsRel, SqRel, Log10, RMSELog, DeltaAccuracy


def evaluate_superdepth_model(model, loss_type, orig_test_ds, preprocess_fn, config, test_ds_dataset_type,
                              depth_factor=10., test_bs=3):
    log_space = is_si_error(loss_type)
    original_shape = [480, 640]
    target_shape = [224, 224]

    censored_depth = 10.

    evaluate_test_ds(model, orig_test_ds, test_ds_dataset_type, original_shape, target_shape, log_space, test_bs,
                     depth_factor)

    conduct_ibims_evaluation(model, config, original_shape, target_shape, preprocess_fn, log_space, depth_factor,
                             censored_depth=censored_depth)

    conduct_diode_evaluation(model, config, target_shape, preprocess_fn, log_space, depth_factor,
                             censored_depth=censored_depth)


def conduct_sunrgbd_evaluation(model, orig_test_ds, original_shape, target_shape, log_space, test_batch_size=3,
                               depth_factor=10.):
    evaluate_test_ds(model, orig_test_ds, Dataset.SUNRGBD, original_shape, target_shape, log_space, test_batch_size,
                     depth_factor)


def evaluate_test_ds(model, orig_test_ds, dataset_type, original_shape, target_shape, log_space, test_batch_size=3,
                     depth_factor=10.):
    metrics = []
    for metric in [RootMeanSquaredError, AbsRel, SqRel, Log10, RMSELog]:
        # Uncensored
        metrics.append(
            construct_test_ds_censored_metric(metric(depth_factor, log_space, upper_bound_pred=False), original_shape,
                                              target_shape, test_batch_size, dataset_type))

    delta_strs = ["1_25", "1_25^2", "1_25^3"]
    for idx, delta in enumerate([1.25, 1.25 ** 2, 1.25 ** 3]):
        # Uncensored
        metrics.append(
            construct_test_ds_censored_metric(
                DeltaAccuracy(delta, depth_factor, log_space, name="delta_acc_{}".format(delta_strs[idx])),
                original_shape, target_shape, test_batch_size, dataset_type))

    model.compile(metrics=metrics)
    results = model.evaluate(orig_test_ds)

    i = 1
    for metric in ["rmse", "absrel", "sqrel", "log10", "rmselog", "delta1_25", "delta1_25_2", "delta1_25_3"]:
        mlflow.log_metric("{}_{}".format(dataset_type, metric), results[i])
        i += 1


def conduct_ibims_evaluation(model, config, original_shape, target_shape, preprocess_fn, log_space, depth_factor=10.,
                             censored_depth=10.):
    # iBims evaluation
    ibims_ds = IbimsTFDataAccessObject(config["DATA"]["IBIMS_ROOT_PATH"], original_shape).get_test_dataset()

    def resize_image_only(image, depth):
        image = tf.ensure_shape(image, original_shape + [3])
        return tf.image.resize(image, target_shape), depth

    ibims_ds = ibims_ds.map(resize_image_only, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ibims_test_batch_size = 1

    ibims_ds = ibims_ds.map(preprocess_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(ibims_test_batch_size)

    ibims_metrics = []
    for metric in [RootMeanSquaredError, AbsRel, SqRel, Log10, RMSELog]:
        # Uncensored
        ibims_metrics.append(
            construct_uncensored_metric(metric(depth_factor, log_space), original_shape, target_shape, 1))

        # Censored
        ibims_metrics.append(
            construct_censored_metric(metric(depth_factor, log_space), censored_depth, original_shape, target_shape, 1))

    delta_strs = ["1_25", "1_25^2", "1_25^3"]
    for idx, delta in enumerate([1.25, 1.25 ** 2, 1.25 ** 3]):
        # Uncensored
        ibims_metrics.append(
            construct_uncensored_metric(
                DeltaAccuracy(delta, depth_factor, log_space, name="delta_acc_{}".format(delta_strs[idx])),
                original_shape, target_shape, 1))

        # Censored
        ibims_metrics.append(
            construct_censored_metric(
                DeltaAccuracy(delta, depth_factor, log_space, name="delta_acc_{}".format(delta_strs[idx])), 10.,
                original_shape, target_shape, 1))

    model.compile(metrics=ibims_metrics)
    ibims_results = model.evaluate(ibims_ds)

    i = 1
    for metric in ["rmse", "absrel", "sqrel", "log10", "rmselog", "delta1_25", "delta1_25_2", "delta1_25_3"]:
        mlflow.log_metric("ibims_{}".format(metric), ibims_results[i])
        mlflow.log_metric("ibims_{}_cen".format(metric), ibims_results[i + 1])
        i += 2


def conduct_diode_evaluation(model, config, target_shape, preprocess_fn, log_space, depth_factor=10.,
                             censored_depth=10.):
    # DIODE evaluation
    diode_ds = DIODETFDataAccessObject(config["DATA"]["DIODE_ROOT_PATH"], [768, 1024]).get_test_dataset()

    def resize_image_only(image, depth):
        image = tf.ensure_shape(image, [768, 1024] + [3])
        return tf.image.resize(image, target_shape), depth

    diode_ds = diode_ds.map(resize_image_only, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    diode_test_batch_size = 1
    diode_ds = diode_ds.map(preprocess_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(diode_test_batch_size)

    diode_metrics = []
    for metric in [RootMeanSquaredError, AbsRel, SqRel, Log10, RMSELog]:
        # Uncensored
        diode_metrics.append(
            construct_uncensored_metric(metric(depth_factor, log_space), [768, 1024], target_shape, 1))

        # Censored
        diode_metrics.append(
            construct_censored_metric(metric(depth_factor, log_space), censored_depth, [768, 1024], target_shape, 1))

    delta_strs = ["1_25", "1_25^2", "1_25^3"]
    for idx, delta in enumerate([1.25, 1.25 ** 2, 1.25 ** 3]):
        # Uncensored
        diode_metrics.append(
            construct_uncensored_metric(
                DeltaAccuracy(delta, depth_factor, log_space, name="delta_acc_{}".format(delta_strs[idx])),
                [768, 1024], target_shape, 1))

        # Censored
        diode_metrics.append(
            construct_censored_metric(
                DeltaAccuracy(delta, depth_factor, log_space, name="delta_acc_{}".format(delta_strs[idx])), 10.,
                [768, 1024], target_shape, 1))

    model.compile(metrics=diode_metrics)
    diode_results = model.evaluate(diode_ds)

    i = 1
    for metric in ["rmse", "absrel", "sqrel", "log10", "rmselog", "delta1_25", "delta1_25_2", "delta1_25_3"]:
        mlflow.log_metric("diode_{}".format(metric), diode_results[i])
        mlflow.log_metric("diode_{}_cen".format(metric), diode_results[i + 1])
        i += 2


def construct_uncensored_metric(int_metric, int_target_shape, model_output_shape, batch_size):
    uncen_metric = DistanceUncensoredError(batch_size, int_metric, name=int_metric.name)
    return ResizedError(int_target_shape, model_output_shape, batch_size, uncen_metric, name=int_metric.name)


def construct_censored_metric(int_metric, max_dist_distance, int_target_shape, model_output_shape, batch_size):
    cen_metric = DistanceCensoredError(batch_size, max_dist_distance, int_metric, name=int_metric.name)
    return ResizedError(int_target_shape, model_output_shape, batch_size, cen_metric,
                        name=cen_metric.name + "_cen")


def construct_test_ds_censored_metric(int_metric, int_target_shape, model_output_shape, batch_size, dataset_type):
    cen_metric = DistanceCensoredError(batch_size, 10., int_metric, name=int_metric.name)
    if dataset_type == Dataset.NYU:
        cropped_metric = CroppingError(NYULargeScaleTFDataAccessObject.EIGEN_CROP, cen_metric, name="crop")
    else:
        cropped_metric = cen_metric
    return ResizedError(int_target_shape, model_output_shape, batch_size, cropped_metric,
                        name=int_metric.name)
