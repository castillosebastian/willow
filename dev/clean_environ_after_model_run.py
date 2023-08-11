import gc
# Clear memory
del model
del tokenizer
del input_ids
del output_ids

import torch
torch.cuda.empty_cache()
gc.collect()