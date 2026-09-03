import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    transformed_name
)

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and label for training."""
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

def _build_keras_model(tf_transform_output, hyperparameters=None):
    """Builds a Keras Deep Neural Network model for binary classification."""
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
        
        # Check hyperparameters from tuner if available
        if hyperparameters and f'embed_dim_{feature}' in hyperparameters.get('values', {}):
            embed_dim = hyperparameters['values'][f'embed_dim_{feature}']
        else:
            embed_dim = 8
            
        cat_column = tf.feature_column.categorical_column_with_identity(key, num_buckets=vocab_size + 1)
        embed_column = tf.feature_column.embedding_column(cat_column, dimension=embed_dim)
        feature_columns.append(embed_column)

    dense_input = tf.keras.layers.DenseFeatures(feature_columns)(input_layers)
    
    # Extract hyperparameters from Tuner output safely
    if hasattr(hyperparameters, 'get'):
        hp = hyperparameters.get('values', hyperparameters)
    elif hasattr(hyperparameters, 'values'):
        hp = hyperparameters.values
    else:
        hp = hyperparameters if isinstance(hyperparameters, dict) else {}
        
    units_1 = hp.get('units_1', 64)
    units_2 = hp.get('units_2', 64)
    dropout_rate = hp.get('dropout_rate', 0.5)
    learning_rate = hp.get('learning_rate', 0.01)

    x = tf.keras.layers.Dense(units_1, activation='relu')(dense_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses raw tf.Example and applies preprocessing."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label key from raw spec for inference serving
        if LABEL_KEY in feature_spec:
            feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        # Remove transformed label key if present
        transformed_label = transformed_name(LABEL_KEY)
        if transformed_label in transformed_features:
            transformed_features.pop(transformed_label)

        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    
    Args:
        fn_args: Holds args used to train model, including transform graph path.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    hparams = fn_args.hyperparameters if fn_args.hyperparameters else None

    model = _build_keras_model(tf_transform_output, hyperparameters=hparams)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch'
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps if fn_args.train_steps else 15,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps if fn_args.eval_steps else 5,
        epochs=10,
        callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
