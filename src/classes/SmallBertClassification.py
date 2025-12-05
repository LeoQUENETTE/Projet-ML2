from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable
from src.classes.SmallBert import SmallBERT
import tensorflow as tf
from tensorflow import keras
@tf.keras.saving.register_keras_serializable()
class SmallBERTForClassification(keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout_rate = 0.3,**kwargs):
        super().__init__(**kwargs)
        self.encoder = SmallBERT(sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers)

        self.dropout = layers.Dropout(dropout_rate)
        self.dense = layers.Dense(128, activation="relu")
        self.batch_norm = layers.BatchNormalization()

        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(64, activation="relu")

        self.output_layer = layers.Dense(4, activation="softmax")
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        mean_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)
        x = tf.concat([mean_pool, max_pool], axis=-1)

        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.batch_norm(x, training=training)

        x = self.dropout2(x, training=training)
        x = self.dense2(x)

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.encoder.pos_embedding.sequence_length,
            "vocab_size": self.encoder.pos_embedding.vocab_size,
            "embed_dim": self.encoder.pos_embedding.embed_dim,
            "num_heads": self.encoder.transformer_blocks[0].att.num_heads,
            "ff_dim": self.encoder.transformer_blocks[0].ffn.layers[0].units,
            "num_layers": len(self.encoder.transformer_blocks),
            "dropout_rate" : self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)