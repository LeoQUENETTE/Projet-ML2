from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable
from SmallBert import SmallBERT
@register_keras_serializable()
class SmallBERTForClassification(Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SmallBERT(sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers)
        self.mlm_head = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        x = self.mlm_head(x)
        return self.mlm_head(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.encoder.pos_embedding.sequence_length,
            "vocab_size": self.encoder.pos_embedding.vocab_size,
            "embed_dim": self.encoder.pos_embedding.embed_dim,
            "num_heads": self.encoder.transformer_blocks[0].att.num_heads,
            "ff_dim": self.encoder.transformer_blocks[0].ffn.layers[0].units,
            "num_layers": len(self.encoder.transformer_blocks),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)