import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def create_x_y(sequence_length : int, max_len : int, df : list[str]):
    tokenizer = Tokenizer(num_words=sequence_length, oov_token="<OOV>") # Tester avec sequence_length = 36 ? (pas satisfaite des datas du prof)
    tokenizer.fit_on_texts(df['caption']) # Apprentissage du vocabulaire
    sequences = tokenizer.texts_to_sequences(df['caption'])

    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y = np.array(df['label'])