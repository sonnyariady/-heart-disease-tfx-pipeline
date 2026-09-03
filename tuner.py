import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any

from transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    transformed_name
)

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', kt.Tuner),
    ('fit_kwargs', Dict[Text, Any])
])

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and label for tuning/training."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def _gzip_reader_fn(filenames):
    """Instantiates a tf.data.TFRecordDataset with GZIP compression."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _build_model(hp, tf_transform_output):
    """Builds a Keras Deep Neural Network model based on hyperparameters."""
    input_layers = {}
    feature_columns = []

    for feature in NUMERICAL_FEATURES:
        key = transformed_name(feature)
        input_layers[key] = tf.keras.layers.Input(shape=(1,), name=key, dtype=tf.float32)
        feature_columns.append(tf.feature_column.numeric_column(key))

    for feature in CATEGORICAL_FEATURES:
        key = transformed_name(feature)
        input_layers[key] = tf.keras.layers.Input(shape=(1,), name=key, dtype=tf.int64)
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature)
        embedding_dim = hp.Int(f'embed_dim_{feature}', min_value=4, max_value=16, step=4)
        cat_column = tf.feature_column.categorical_column_with_identity(key, num_buckets=vocab_size + 1)
        embed_column = tf.feature_column.embedding_column(cat_column, dimension=embedding_dim)
        feature_columns.append(embed_column)

    dense_input = tf.keras.layers.DenseFeatures(feature_columns)(input_layers)
    
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])

    x = tf.keras.layers.Dense(units_1, activation='relu')(dense_input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def tuner_fn(fn_args):
    """Build the tuner using KerasTuner.
    
    Args:
        fn_args: Holds args used to tune models, including transform output.
    Returns:
        A NamedTuple containing the tuner and fit_kwargs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: _build_model(hp, tf_transform_output),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=3,
        directory=fn_args.working_dir,
        project_name='heart_disease_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'epochs': 5,
            'steps_per_epoch': 10,
            'validation_steps': 5
        }
    )
