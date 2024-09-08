import tensorflow as tf
from config.constants import DIRECTORY, IMAGE_SIZE, BATCH_SIZE, SHUFFLE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SHUFFLE_SIZE, SEED

def load_and_prepare_data():
    """Loads and prepares the dataset from the given directory."""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DIRECTORY,
        shuffle=SHUFFLE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    return dataset

def get_dataset_partitions_tf(ds):
    """Splits the dataset into train, validation, and test sets."""
    assert (TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) == 1, "Splits must sum to 1"
    
    ds_size = len(ds)
    if SHUFFLE:
        ds = ds.shuffle(SHUFFLE_SIZE, seed=SEED)

    train_size = int(TRAIN_SPLIT * ds_size)
    val_size = int(VAL_SPLIT * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds

def prepare_datasets_for_training(train_ds, val_ds, test_ds):
    """Prepares datasets for training with caching and prefetching."""
    train_ds = train_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_ds, val_ds, test_ds