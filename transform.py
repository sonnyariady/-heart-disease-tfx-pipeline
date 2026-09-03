import tensorflow as tf
import tensorflow_transform as tft

# Define Feature Keys
NUMERICAL_FEATURES = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
]

CATEGORICAL_FEATURES = [
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
]

LABEL_KEY = 'target'

def transformed_name(key: str) -> str:
    """Utility function to rename transformed feature key."""
    return key + '_xf'

def preprocessing_fn(inputs: dict) -> dict:
    """Preprocess input features using TensorFlow Transform.
    
    Args:
        inputs: map from feature keys to raw tf.Tensor.
    Returns:
        map from feature keys to transformed tf.Tensor.
    """
    outputs = {}
    
    # Standardize numerical features using z-score normalization
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(inputs[feature], tf.float32)
        )
        
    # Convert categorical features to integer vocabulary indices / cast as int64
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            tf.cast(inputs[feature], tf.string)
            if inputs[feature].dtype == tf.string
            else tf.strings.as_string(inputs[feature]),
            vocab_filename=feature
        )
        
    # Target feature remains unchanged or cast to int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
