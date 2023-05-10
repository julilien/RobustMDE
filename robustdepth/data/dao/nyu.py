import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from robustdepth.data.data_meta import TFDataAccessObject


class NYULargeScaleTFDataAccessObject(TFDataAccessObject):
    # Cropping indices as proposed by Eigen (y_start, y_end, x_start, x_end)
    EIGEN_CROP = np.array([45, 471, 41, 601])

    def __init__(self, root_path, target_shape, config, seed, augmentation=True, provide_val_split=True,
                 val_split_size=0.05, sample_subset=None):
        self.target_shape = target_shape
        assert len(target_shape) >= 2, "The target shape must at least contain the height and width."
        assert target_shape[0] <= 480 and target_shape[1] <= 640, "The target shape must be at maximum 480 x 640 " \
                                                                  "(height x width)."
        self.config = config
        self.seed = seed
        self.augmentation = augmentation

        train_paths = open(os.path.join(root_path, "data/nyu2_train.csv"), 'r').read()
        train_pairs = list((row.split(',') for row in train_paths.split('\n') if len(row) > 0))
        self.train_pairs = shuffle(train_pairs, random_state=seed)

        if sample_subset is not None and sample_subset > 0:
            num_examples = len(self.train_pairs)
            assert 1 < sample_subset < num_examples, "Sample subsize must be between 1 and {}".format(num_examples)

            self.train_pairs = self.train_pairs[:sample_subset]

        if provide_val_split:
            self.train_pairs, self.val_pairs = train_test_split(self.train_pairs, test_size=val_split_size,
                                                                random_state=seed)
        else:
            self.val_pairs = None

    def _parse_function(self, filename, label):
        image_decoded = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(filename)),
                                        [self.target_shape[0], self.target_shape[1]]) / 255.
        depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                        [self.target_shape[0], self.target_shape[1]])

        # Format
        rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        depth = tf.image.convert_image_dtype(depth_resized / 255., dtype=tf.float32)

        # Normalize the depth values (in cm)
        # Raw depths are given by depth * 10
        depth = tf.clip_by_value(depth * 1000, 10, 1000) / 1000

        return rgb, depth

    def construct_ds_from_pairs(self, pairs):
        image_paths = [os.path.join(self.config["DATA"]["NYU_LARGE_ROOT_PATH"], i[0]) for i in pairs]
        gt_paths = [os.path.join(self.config["DATA"]["NYU_LARGE_ROOT_PATH"], i[1]) for i in pairs]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, gt_paths))

        return dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_training_dataset(self):
        train_ds = self.construct_ds_from_pairs(self.train_pairs)
        if self.augmentation:
            train_ds = train_ds.map(self.augment_image_depth_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return train_ds

    def get_validation_dataset(self):
        if self.val_pairs is None:
            raise ValueError("A validation dataset can not be provided if it has not been considered at DAO "
                             "initialization time.")
        else:
            return self.construct_ds_from_pairs(self.val_pairs)

    def get_test_dataset(self):
        file_pattern_images = os.path.join(self.config["DATA"]["NYU_LARGE_ROOT_PATH"], 'nyu_eigen_test_split/rgb_*.jpg')

        # This is necessary to keep the same order of the files
        file_name_images = [s for s in
                            tf.data.Dataset.list_files(file_pattern_images, shuffle=False).as_numpy_iterator()]
        file_name_depths = [s.replace(b'.jpg', b'.png') for s in file_name_images]
        file_name_depths = [s.replace(b'rgb', b'sync_depth') for s in file_name_depths]

        def process_image(x):
            image = tf.cast(tf.image.decode_jpeg(tf.io.read_file(x)), dtype=tf.float32) / 255.
            return tf.image.resize(image, self.target_shape[:2])

        images_ds = tf.data.Dataset.from_tensor_slices(file_name_images).map(process_image,
                                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def process_depth(x):
            depth = tf.cast(tf.image.decode_png(tf.io.read_file(x), channels=0, dtype=tf.uint16),
                            dtype=tf.float32) / 1000.
            return depth

        gts_ds = tf.data.Dataset.from_tensor_slices(file_name_depths).map(process_depth)

        return tf.data.Dataset.zip((images_ds, gts_ds))
