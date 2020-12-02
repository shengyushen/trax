#### Copyright 2020 Google LLC.

# Licensed under the Apache License, Version 2.0 (the \"License\")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Reformer: Text Generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/text_generation.ipynb)

#This notebook was designed to run on TPU.
# To use TPUs in Colab, click \"Runtime\" on the main menu bar and select Change runtime type. Set \"TPU\" as the hardware accelerator.

# Install JAX.
# !pip install --upgrade jax
# !pip install --upgrade jaxlib
# !pip install --upgrade trax
# Make sure the Colab Runtime is set to Accelerator: TPU.
import requests
import os
# I dont have TPU
# if 'TPU_DRIVER_MODE' not in globals():
#   url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver0.1-dev20191206'
#   resp = requests.post(url)
#   TPU_DRIVER_MODE = 1

# The following is required to use TPU Driver as JAX's backend.
from jax.config import config
# I dont have tpu
# config.FLAGS.jax_xla_backend = "tpu_driver"
# config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
print(config.FLAGS.jax_backend_target)

# !pip install --upgrade -q sentencepiece
# !pip install --upgrade -q gin 

from tensorflow.compat.v1.io.gfile import GFile
import gin
import os
import jax
import trax
from trax.data import inputs

import numpy as np
import jax.numpy as jnp

from scipy.special import softmax

from sentencepiece import SentencePieceProcessor

# Setting up data and model

# In this notebook, we'll be pushing the limits of just how many tokens we can fit on a single TPU device. The TPUs available in Colab have 8GB of memory per core, and 8 cores. We will set up a Reformer model that can fit a copy of \"Crime and Punishment\" on *each* of the 8 TPU cores (over 500,000 tokens per 8GB of memory).

# Import a copy of \"Crime and Punishment\ by Fyodor Dostoevsky
#with GFile('gs://trax-ml/reformer/crime-and-punishment-2554.txt') as f:
with GFile('trax-ml/reformer/crime-and-punishment-2554.txt') as f:
  text = f.read()

# The file read above includes metadata and licensing information.
# For training our language model, we will only use the actual novel text.
start = text.find('CRIME AND PUNISHMENT')  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip translator preface
end = text.rfind('End of Project')  # skip extra text at the end
text = text[start:end].strip()

# Load a BPE vocabulaary with 320 types. This mostly consists of single letters
# and pairs of letters, but it has some common words and word pieces, too.
# !gsutil cp gs://trax-ml/reformer/cp.320.* .

TOKENIZER = SentencePieceProcessor()
#TOKENIZER.load('cp.320.model')
TOKENIZER.load('trax-ml/reformer/cp.320.model')

# Tokenize
IDS = TOKENIZER.EncodeAsIds(text)
IDS = np.asarray(IDS, dtype=np.int32)
PAD_AMOUNT = 512 * 1024 - len(IDS)
print("Number of tokens:", IDS.shape[0])

# As we see above, \"Crime and Punishment\" has just over half a million tokens with the BPE vocabulary we have selected.

# Normally we would have a dataset with many examples, but for this demonstration we fit a language model on the single novel only. We don't want the model to just memorize the dataset by encoding the words in its position embeddings, so at each training iteration we will randomly select how much padding to put before the text vs. after it.

# We have 8 TPU cores, so we will separately randomize the amount of padding for each core."

# Set up the data pipeline.
def my_inputs(n_devices):
  print("SSY n_devices "+str(n_devices))
  while True:
    inputs = []
    mask = []
    pad_amounts = np.random.choice(PAD_AMOUNT, n_devices)
    for i in range(n_devices):
      inputs.append(np.pad(IDS, (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                            mode='constant'))
      mask.append(np.pad(np.ones_like(IDS, dtype=np.float32),
                          (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                          mode='constant'))
    inputs = np.stack(inputs)
    mask = np.stack(mask)
    yield (inputs, inputs, mask)
# SSY /opt/conda/lib/python3.7/site-packages/jax/lib/xla_bridge.py some where
# SSY /opt/conda/lib/python3.7/site-packages/trax/fastmath/ops.py
#print("calling my_inputs before parse_config")
#print("(device count, tokens per device) =", next(my_inputs(trax.fastmath.device_count()))[0].shape)

# Configure hyperparameters.
gin.parse_config("""
#import trax.layers
#import trax.models
#import trax.optimizers
#import trax.data.inputs
#import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
# Our model will have 6 layers, alternating between the LSH attention proposed
# in the Reformer paper and local attention within a certain context window.
# SSY interleaved LSH and self attention? which means we still need to depend on self attention
n_layers = 6
attn_type = [
  @trax.layers.SelfAttention,
  @LSHSelfAttention,  
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  ]
share_qk = False  # LSH attention ignores this flag and always shares q & k
n_heads = 2
attn_kv = 64
dropout = 0.05
n_tokens = 524288

# Parameters for multifactor:
# ==============================================================================
multifactor.constant = 0.01
multifactor.factors = 'constant * linear_warmup * cosine_decay'
multifactor.warmup_steps = 100
multifactor.steps_per_cycle = 900

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate=0.0
Adam.b1 = 0.86
Adam.b2 = 0.92
Adam.eps = 1e-9

# Parameters for SelfAttention:
# ==============================================================================
trax.layers.SelfAttention.attention_dropout = 0.05
trax.layers.SelfAttention.chunk_len = 64
trax.layers.SelfAttention.n_chunks_before = 1
trax.layers.SelfAttention.n_parallel_heads = 1

# Parameters for LSHSelfAttention:
# ==============================================================================
LSHSelfAttention.attention_dropout = 0.0
LSHSelfAttention.chunk_len = 64
LSHSelfAttention.n_buckets = [64, 128]
LSHSelfAttention.n_chunks_after = 0
LSHSelfAttention.n_chunks_before = 1
LSHSelfAttention.n_hashes = 1
LSHSelfAttention.n_parallel_heads = 1
LSHSelfAttention.predict_drop_len = 128
LSHSelfAttention.predict_mem_len = 1024

# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.attention_type = %attn_type
ReformerLM.d_attention_key = %attn_kv
ReformerLM.d_attention_value = %attn_kv
ReformerLM.d_model = 256
ReformerLM.d_ff = 512
ReformerLM.dropout = %dropout
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = %n_tokens
ReformerLM.mode = 'train'
ReformerLM.n_heads = %n_heads
ReformerLM.n_layers = %n_layers
ReformerLM.vocab_size = 320
ReformerLM.axial_pos_shape = (512, 1024)
ReformerLM.d_axial_pos_embs= (64, 192)
""")

print("makeing sure above setting is correctly handled by gin.parse_config")
print(gin.query_parameter("ReformerLM.axial_pos_shape"))

# Set up a Trainer.
output_dir = os.path.expanduser('~/train_dir/')
# !rm -f ~/train_dir/model.pkl.gz  # Remove old model

trainer = trax.supervised.Trainer(
    model=trax.models.ReformerLM,
    loss_fn=trax.layers.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam,
    lr_schedule=trax.lr.multifactor(),
    inputs=trax.data.inputs.Inputs(my_inputs), # SSY Inputs is class defined in /opt/conda/lib/python3.7/site-packages/trax/data/inputs.py
    output_dir=output_dir)

# Run one training step, to make sure the model fits in memory.
# The first time trainer.train_epoch is called, it will JIT the entire network
# architecture, which takes around 2 minutes. The JIT-compiled model is saved
# so subsequent runs will be much faster than the first.
trainer.train_epoch(n_steps=1, n_eval_steps=1)

# Train for 600 steps total
# The first ~20 steps are slow to run, but after that it reaches steady-state
# speed. This will take at least 30 minutes to run to completion, but can safely
# be interrupted by selecting \"Runtime > Interrupt Execution\" from the menu.
# The language model won't be exceptionally good when trained for just a few
# steps and with minimal regularization. However, we can still sample from it to
# see what it learns.
trainer.train_epoch(n_steps=9, n_eval_steps=1)
#for _ in range(59):
for i in range(59):
  print("iteration "+str(i))
  trainer.train_epoch(n_steps=10, n_eval_steps=1)

## Sample from the model"

# As we report in the Reformer paper, increasing the number of hashing rounds
# helps with quality. We can even increase the number of hashing rounds at
# evaluation time only.

gin.parse_config("LSHSelfAttention.n_hashes = 4") # SSY hash 4 times

# Load the trained Reformer in 'predict' mode
model = trax.models.ReformerLM(mode='predict')
model.init_from_file(os.path.join(output_dir,'model.pkl.gz'),
                     weights_only=True)

# Sample from ReformerLM
output_token_ids = trax.supervised.decoding.autoregressive_sample(
    model, temperature=0.0)

# Decode token IDs
# Reformer outputed a batch with one item, we access it using [0]
# tolist() converts from int64 to int, the type SentencePiece expects
TOKENIZER.DecodeIds(output_token_ids[0].tolist())

