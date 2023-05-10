import click
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
import mlflow

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from kerastuner import Objective, RandomSearch

from tensorflow.keras.experimental import CosineDecay

from robustdepth.data.dao.ibims import IbimsTFDataAccessObject
from robustdepth.data.dao.nyu import NYULargeScaleTFDataAccessObject
from robustdepth.data.dao.sunrgbd import SunRGBDDataAccessObject
from robustdepth.data.io_utils import get_dataset_type_by_name, Dataset
from robustdepth.eval.superdepth_eval import evaluate_superdepth_model
from robustdepth.losses.barron_loss import BarronLoss
from robustdepth.losses.eps_sens_loss import EpsSensL1Loss, EpsSensL2Loss
from robustdepth.losses.fosl_trapezoidal import FOSLL1Trapezoidal
from robustdepth.losses.huber import BerhuLoss, HuberLoss, RuberLoss
from robustdepth.losses.losses_meta import get_loss_type_by_name, DepthLossType, \
    is_trapezoidal_fosl, is_si_error
from robustdepth.losses.si_error import ScaledSIError
from robustdepth.losses.trim_loss import TrimLoss
from robustdepth.losses.weighted_l2 import WeightedL2Loss
from robustdepth.metrics.depth_metrics import DistanceCensoredError, RootMeanSquaredError
from robustdepth.models.depth.unet_model import get_unet_model
from robustdepth.models.models_meta import ModelParameters, get_model_type_by_name
from robustdepth.util.env import init_env


@click.command()
@click.option('--model_name', default='ff_effnet', help='Backbone model',
              type=click.Choice(['ff_effnet', 'ff_effnetb5'], case_sensitive=False))
@click.option('--epochs', default=25)
@click.option('--batch_size', default=4)
@click.option('--seed', default=0)
@click.option('--autolog_freq', default=1)
@click.option('--cluster_job', default=False, type=click.BOOL)
@click.option('--dataset', default="NYU", type=click.Choice(['NYU', 'SUNRGBD']))
@click.option('--loss_name', default='l1', help='Loss function to be used',
              type=click.Choice(
                  ['l1', 'l2', 'berhu', 'huber', 'fosll1_trapezoidal', 'scaled_si_error', 'weighted_l2', 'barron',
                   'eps_sens_l1', 'eps_sens_l2', 'trim', 'ruber'],
                  case_sensitive=False))
@click.option('--augmentation', default=True, type=click.BOOL)
@click.option('--project_name', default="", type=click.STRING)
@click.option('--max_trials', type=click.INT, default=20)
@click.option('--num_data_points', type=click.INT, default=10000)
@click.option('--bo_directory', type=click.STRING, default="bo", help="Directory relative to current directory")
@click.option('--experiment_name', type=click.STRING, default="RobustDepthRegression",
              help="Mlflow experiment name for the stored experiment.")
@click.option('--noise_level', type=click.FLOAT, default=0.0)
def perform_robustness_experiment(model_name, epochs, batch_size, seed, autolog_freq, cluster_job, dataset, loss_name,
                                  augmentation, project_name, max_trials, num_data_points, bo_directory,
                                  experiment_name, noise_level):
    config = init_env(autolog_freq=autolog_freq, seed=seed, experiment_name=experiment_name, log_mlflow_models=False)

    max_depth_factor = 10.

    with mlflow.start_run():
        # Determine model, dataset and loss types
        model_type = get_model_type_by_name(model_name)
        dataset_type = get_dataset_type_by_name(dataset)
        loss_type = get_loss_type_by_name(loss_name)

        if project_name == "":
            loss_str = loss_name
            project_name = "rs_run_{}_{}_{}_{}_{}".format(dataset, loss_str, num_data_points, seed,
                                                          int(noise_level * 100))

        # Run meta information
        model_params = ModelParameters()
        model_params.set_parameter("model_type", model_type)
        model_params.set_parameter("dataset", dataset_type)
        model_params.set_parameter("epochs", epochs)
        model_params.set_parameter("batch_size", batch_size)
        model_params.set_parameter("prediction_batch_size", batch_size)
        model_params.set_parameter("seed", seed)
        model_params.set_parameter('loss_type', loss_type)
        model_params.set_parameter('augmentation', augmentation)
        model_params.set_parameter('project_name', project_name)
        model_params.set_parameter('max_trials', max_trials)
        model_params.set_parameter('num_data_points', num_data_points)
        model_params.set_parameter('bo_directory', bo_directory)
        model_params.set_parameter('noise_level', noise_level)

        model_input_shape = [224, 224, 3]

        # Get model
        model, norm, preprocess_fn = get_unet_model(model_params, model_input_shape)

        if not cluster_job:
            model.summary()

        test_batch_size = 3
        if dataset_type == Dataset.NYU:
            dao = NYULargeScaleTFDataAccessObject(config["DATA"]["NYU_LARGE_ROOT_PATH"], model_input_shape, config,
                                                  seed,
                                                  True, provide_val_split=False, sample_subset=num_data_points)
        elif dataset_type == Dataset.SUNRGBD:
            dao = SunRGBDDataAccessObject(config["DATA"]["SUNRGBD_ROOT_PATH"], model_input_shape, seed,
                                          val_split_size=0.0, sample_subset=num_data_points)
        else:
            raise ValueError("Unsupported data set type {}.".format(dataset_type))

        train_ds = dao.get_training_dataset().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        def introduce_noise(image, depth):
            std_dev = 0.01 * tf.math.pow(depth * 10., 2.) + tf.constant(noise_level, dtype=tf.float32)

            return image, keras_backend.clip(tf.random.normal(shape=tf.shape(depth), mean=depth,
                                                              stddev=std_dev, dtype=tf.float32, seed=seed), 1e-3, None)

        if noise_level > 0.0:
            train_ds = train_ds.map(introduce_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        test_ds = dao.get_test_dataset().batch(test_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # iBims validation set
        ibims_ds = IbimsTFDataAccessObject(config["DATA"]["IBIMS_ROOT_PATH"], model_input_shape).get_test_dataset()
        ibims_test_batch_size = batch_size

        def transform_depth(img, depth):
            return img, depth / 10.

        val_ds = ibims_ds.map(transform_depth, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            ibims_test_batch_size, drop_remainder=True)

        # Preprocess datasets (e.g., as assumed for a specific pretrained encoder model)
        train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def build_model(hp, learning_rate=None, si_error_lambda=None, fosl_core_width=None,
                        fosl_support_width=None, si_error_alpha=None, epsilon=None,
                        barron_alpha=None, barron_scale=None, uber_c=None, trim_c=None):
            if learning_rate is None:
                learning_rate = hp.Float('learning_rate', max_value=1e-1, min_value=1e-4, sampling="log")

            # Compile model
            if loss_type == DepthLossType.MEAN_ABS_ERROR:
                loss_fn = MeanAbsoluteError()
            elif loss_type == DepthLossType.MEAN_SQR_ERROR:
                loss_fn = MeanSquaredError()
            elif loss_type == DepthLossType.WEIGHTED_L2:
                loss_fn = WeightedL2Loss()
            elif loss_type == DepthLossType.TRIM:
                if trim_c is None:
                    trim_c = hp.Float('trim_c', min_value=0.1, max_value=0.9)

                loss_fn = TrimLoss(batch_size, c_param=trim_c)
            elif loss_type == DepthLossType.SCALED_SI_ERROR:
                if si_error_lambda is None:
                    si_error_lambda = hp.Float('si_error_lambda', min_value=0.1, max_value=0.9)
                if si_error_alpha is None:
                    si_error_alpha = hp.Float('si_error_alpha', min_value=0.1, max_value=25.)

                loss_fn = ScaledSIError(lambda_val=si_error_lambda, alpha_val=si_error_alpha)
            elif loss_type == DepthLossType.BERHU:
                if uber_c is None:
                    uber_c = hp.Float('uber_c', min_value=0.1, max_value=0.9)

                loss_fn = BerhuLoss(uber_c)
            elif loss_type == DepthLossType.HUBER:
                if uber_c is None:
                    uber_c = hp.Float('uber_c', min_value=0.1, max_value=0.9)

                loss_fn = HuberLoss(uber_c)
            elif loss_type == DepthLossType.RUBER:
                if uber_c is None:
                    uber_c = hp.Float('uber_c', min_value=0.1, max_value=0.9)

                loss_fn = RuberLoss(uber_c)
            elif loss_type == DepthLossType.BARRON:
                if barron_alpha is None:
                    barron_alpha = hp.Float('barron_alpha', min_value=0.0, max_value=2.)
                if barron_scale is None:
                    barron_scale = hp.Float('barron_scale', min_value=0.1, max_value=50.)

                loss_fn = BarronLoss(barron_alpha, barron_scale)
            elif loss_type == DepthLossType.EPS_SENS_L1:
                if epsilon is None:
                    epsilon = hp.Float('epsilon', min_value=0.0, max_value=0.25)

                loss_fn = EpsSensL1Loss(batch_size, epsilon)
            elif loss_type == DepthLossType.EPS_SENS_L2:
                if epsilon is None:
                    epsilon = hp.Float('epsilon', min_value=0.0, max_value=0.25)

                loss_fn = EpsSensL2Loss(batch_size, epsilon)
            elif is_trapezoidal_fosl(loss_type):
                if fosl_core_width is None:
                    fosl_core_width = hp.Float('fosl_core_width', min_value=0.0, max_value=0.25)
                if fosl_support_width is None:
                    fosl_support_width = hp.Float('fosl_support_width', min_value=0.0, max_value=2.0)

                def core_fn(x):
                    return tf.cast(float(fosl_core_width), tf.float32)

                def support_fn(x):
                    return tf.cast(float(fosl_support_width), tf.float32)

                core_fns = [core_fn, core_fn]
                support_fns = [support_fn, support_fn]

                loss_fn = FOSLL1Trapezoidal(max_depth_factor, core_fns, support_fns)
            else:
                raise NotImplementedError("Unsupported loss function type '{}'.".format(loss_type))

            lr_schedule = CosineDecay(float(learning_rate), epochs, alpha=0.01)

            optimizer = Adam(learning_rate=lr_schedule, amsgrad=True)

            cen_metric = DistanceCensoredError(batch_size, 10.,
                                               RootMeanSquaredError(max_depth_factor, log_space=is_si_error(loss_type),
                                                                    name="root_mean_squared_error",
                                                                    rescale_target=True),
                                               name="root_mean_squared_error")
            metrics = [cen_metric]

            model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

            model_params.log_parameters()

            return model

        objective_loss = "val_root_mean_squared_error"

        tuner = RandomSearch(build_model, Objective(objective_loss, direction="min"), max_trials=max_trials, seed=seed,
                             directory=bo_directory, project_name=project_name)

        tuner.search_space_summary()
        callbacks = [TerminateOnNaN(), EarlyStopping(patience=15)]
        tuner.search(x=train_ds, epochs=model_params.get_parameter("epochs"), callbacks=callbacks,
                     validation_data=val_ds)

        if not cluster_job:
            tuner.results_summary()
        results_txt = "Final results:\n=========\n"
        for idx, trial in enumerate(tuner.oracle.get_best_trials(3)):
            hyp = trial.hyperparameters
            results_txt += "### Trial #{} (ID {}) ###\n\n".format(idx, trial.trial_id)
            for hp, value in hyp.values.items():
                results_txt += "{}: {}\n".format(hp, value)

            results_txt += "\nScore: {}\n".format(trial.score)
            results_txt += "\n\n"
        if not cluster_job:
            print(results_txt)
        mlflow.set_tag("mlflow.note.content", results_txt)
        if not cluster_job:
            # Attempts to write to ./artifacts (for which we cannot assume write permission)
            mlflow.log_text(results_txt, "best_runs.txt")

        mlflow.log_param("best_trial_id", tuner.oracle.get_best_trials(1)[0].trial_id)

        model = tuner.get_best_models(1)[0]

        evaluate_superdepth_model(model, loss_type, test_ds, preprocess_fn, config, dataset_type, max_depth_factor,
                                  test_bs=test_batch_size)


if __name__ == "__main__":
    perform_robustness_experiment()
