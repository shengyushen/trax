import os
import numpy as np
import trax

# Create a Transformer model.
# Pre-trained model config in gs://trax-ml/models/translation/ende_wmt32k.gin
model = trax.models.Transformer( # SSY trax/models/transformer.py
    input_vocab_size=33300,
    d_model=512, d_ff=2048,
    n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
    max_len=2048, mode='predict') # SSY only using such short length?

# Initialize using pre-trained weights.
#model.init_from_file('gs://trax-ml/models/translation/ende_wmt32k.pkl.gz',
model.init_from_file('./trax-ml/models/translation/ende_wmt32k.pkl.gz',
                     weights_only=True)

# Tokenize a sentence.
#sentence = 'It is nice to learn new things today!'
sentence = 'shengyu shen is an engineer'
# SSY trax/data/tf_inputs.py
tokenized = list(trax.data.tokenize(iter([sentence]),  # Operates on streams.
                                    # vocab_dir='gs://trax-ml/vocabs/',
                                    vocab_dir='./trax-ml/vocabs/',
                                    vocab_file='ende_32k.subword'))[0]

# Decode from the Transformer.
tokenized = tokenized[None, :]  # Add batch dimension.
tokenized_translation = trax.supervised.decoding.autoregressive_sample(
    model, tokenized, temperature=0.0)  # Higher temperature: more diverse results.

# De-tokenize,
tokenized_translation = tokenized_translation[0][:-1]  # Remove batch and EOS.
translation = trax.data.detokenize(tokenized_translation,
                                   # vocab_dir='gs://trax-ml/vocabs/',
                                   vocab_dir='./trax-ml/vocabs/',
                                   vocab_file='ende_32k.subword')
print(translation)
