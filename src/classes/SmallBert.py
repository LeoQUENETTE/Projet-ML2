from src.classes.UsefullClasses import PositionalEmbedding
from TransformerBlock import TransformerBlock
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
@register_keras_serializable()
class SmallBERT(keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, training=False):
        x = self.pos_embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.pos_embedding.sequence_length,
            "vocab_size": self.pos_embedding.vocab_size,
            "embed_dim": self.pos_embedding.embed_dim,
            "num_heads": self.transformer_blocks[0].att.num_heads,
            "ff_dim": self.transformer_blocks[0].ffn.layers[0].units,
            "num_layers": len(self.transformer_blocks),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)