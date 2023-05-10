import glob
import os
from sklearn.utils import shuffle
import tensorflow as tf

from robustdepth.data.data_meta import TFDataAccessObject


class SunRGBDDataAccessObject(TFDataAccessObject):
    IMAGE_DIR_TAG = "image"
    DEPTH_DIR_TAG = "depth_bfx"
    DEPTH_DIR_TAG_TEST = "depth"

    def __init__(self, root_path, target_shape, seed, val_split_size=0.15, sample_subset=None, augmentation=True):
        self.root_path = root_path
        self.target_shape = target_shape
        self.seed = seed
        self.augmentation = augmentation

        self.image_files, self.gt_files = self._retrieve_image_and_gt_file_paths(
            "SUNRGBD/**/" + SunRGBDDataAccessObject.IMAGE_DIR_TAG + "/*.jpg", SunRGBDDataAccessObject.DEPTH_DIR_TAG)

        self.image_files, self.gt_files = shuffle(self.image_files, self.gt_files, random_state=self.seed)

        if sample_subset is not None and sample_subset > 0:
            num_examples = len(self.image_files)
            assert 1 < sample_subset < num_examples, "Sample subsize must be between 1 and {}".format(num_examples)

            self.image_files = self.image_files[:sample_subset]
            self.gt_files = self.gt_files[:sample_subset]

        num_training_examples = int((1. - val_split_size) * len(self.image_files))
        self.train_images = self.image_files[:num_training_examples]
        self.train_depths = self.gt_files[:num_training_examples]

        self.val_images = self.image_files[num_training_examples:]
        self.val_depths = self.gt_files[num_training_examples:]

    def _retrieve_image_and_gt_file_paths(self, base_path, depth_dir_tag):
        img_file_paths = glob.glob(os.path.join(self.root_path, base_path), recursive=True)
        gt_file_paths = [glob.glob(
            s.replace(s.split("/")[-1], "*.png").replace("/" + SunRGBDDataAccessObject.IMAGE_DIR_TAG + "/",
                                                         "/" + depth_dir_tag + "/"))[0] for s in
                         img_file_paths]

        return img_file_paths, gt_file_paths

    def _parse_function(self, filename, label):
        image_decoded = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(filename)),
                                        [self.target_shape[0], self.target_shape[1]]) / 255.

        raw_depth = tf.image.decode_png(tf.io.read_file(label), dtype=tf.uint16)
        depth_in_cm = tf.cast(tf.bitwise.bitwise_or(tf.bitwise.left_shift(raw_depth, 13),
                                                    tf.bitwise.right_shift(raw_depth, 3)), dtype=tf.float32) / 10000.
        depth_resized = tf.image.resize(depth_in_cm, [self.target_shape[0], self.target_shape[1]])

        # Format
        rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        depth = tf.squeeze(depth_resized)

        return rgb, depth

    def get_training_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.train_images, dtype=tf.string),
                                                      tf.convert_to_tensor(self.train_depths, dtype=tf.string)))
        train_ds = dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.augmentation:
            train_ds = train_ds.map(self.augment_image_depth_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return train_ds

    def get_validation_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.val_images, dtype=tf.string),
                                                      tf.convert_to_tensor(self.val_depths, dtype=tf.string)))
        return dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_test_dataset(self):
        test_image_paths, test_depth_paths = self._retrieve_image_and_gt_file_paths(
            "SUNRGBDv2Test/**/" + SunRGBDDataAccessObject.IMAGE_DIR_TAG + "/*.jpg",
            SunRGBDDataAccessObject.DEPTH_DIR_TAG_TEST)

        # Remove invalid image
        img_to_remove = os.path.join(self.root_path,
                                     "SUNRGBDv2Test/black_batch1/2016-02-15T13.43.10.008-0000053860/image/2016-02-15T13.43.10.008-0000053860.jpg")
        if img_to_remove in test_image_paths:
            test_image_paths.remove(img_to_remove)
        depth_to_remove = os.path.join(self.root_path,
                                       "SUNRGBDv2Test/black_batch1/2016-02-15T13.43.10.008-0000053860/depth/2016-02-15T13.43.10.008-0000053860.png")
        if depth_to_remove in test_depth_paths:
            test_depth_paths.remove(depth_to_remove)
        assert len(test_image_paths) == len(test_depth_paths), "Number of images must match the number of depths!"

        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(test_image_paths, dtype=tf.string),
                                                      tf.convert_to_tensor(test_depth_paths, dtype=tf.string)))

        # Test to m
        def depth_to_m(image, depth):
            return image, depth * 10.

        return dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(depth_to_m,
                                                                                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
