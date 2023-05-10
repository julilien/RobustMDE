import abc
import tensorflow as tf

from robustdepth.data.preprocessing.augmentation import flip_image_horizontally, color_augmentation, swap_color_channels

TESTING_ONLY_STR = "{} is intended to be used for testing only and, thus, this DAO does not provide a {} set."


class DataProvider(abc.ABC):
    """
    Abstract base class abstracting the construction of datasets fed to the model.
    """

    def __init__(self, model_params):
        self.model_params = model_params

    @abc.abstractmethod
    def provide_train_dataset(self, X, y, access_ds=None):
        pass

    @abc.abstractmethod
    def provide_val_dataset(self, X, y, access_ds=None):
        pass


class TFDatasetDataProvider(abc.ABC):
    """
    Abstract base class abstracting the construction of datasets fed to the model based on TF datasets.
    """

    def __init__(self, model_params):
        self.model_params = model_params

    @abc.abstractmethod
    def provide_train_dataset(self, base_ds, base_ds_gts=None):
        pass

    @abc.abstractmethod
    def provide_val_dataset(self, base_ds, base_ds_gts=None):
        pass


class DataAccessObject(abc.ABC):
    """
    Abstract base class providing access to a concrete data set, i.e., it abstracts dealing with a specific dataset
    storage in a certain way (directory pattern, database structure, ...).
    """

    @abc.abstractmethod
    def get_images(self):
        pass

    @abc.abstractmethod
    def get_labels(self):
        pass

    @abc.abstractmethod
    def get_suggested_set_indices(self):
        pass


class TFDataAccessObject(abc.ABC):
    @abc.abstractmethod
    def get_training_dataset(self):
        pass

    @abc.abstractmethod
    def get_validation_dataset(self):
        pass

    @abc.abstractmethod
    def get_test_dataset(self):
        pass

    @staticmethod
    def read_file_png(file_path, num_channels=3):
        return tf.cast(tf.image.decode_png(tf.io.read_file(file_path), channels=num_channels), dtype=tf.float32) / 255.

    @staticmethod
    def read_file_jpg(file_path, num_channels=3):
        return tf.cast(tf.image.decode_jpeg(tf.io.read_file(file_path), channels=num_channels), dtype=tf.float32) / 255.

    @staticmethod
    def augment_image_depth_pair(loc_img, loc_gt):
        loc_img, loc_gt = flip_image_horizontally(loc_img, loc_gt)
        loc_img, loc_gt = color_augmentation(loc_img, loc_gt)
        loc_img, loc_gt = swap_color_channels(loc_img, loc_gt)

        return loc_img, loc_gt
