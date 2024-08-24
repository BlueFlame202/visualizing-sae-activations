"""
download_oai_sae.py

This file allows us to cache features. 
"""

import sys
sys.path.append('../') # to allow for better organization

import torch
from nnsight import LanguageModel

from settings import settings

from deps.sae_auto_interp.autoencoders import load_autoencoders
from deps.sae_auto_interp.features import FeatureCache
from deps.sae_auto_interp.utils import load_tokenized_data

import os

print("Starting. Please wait...")
feature_path = settings["feature_dir"]
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print('Model done...')

oai_sae_path = os.path.join(settings["oai_autoencoder_dir"], "gpt2-small/resid_post_mlp_v5_32k/") # change this to the path you would like to use if needed

layers = list(range(0,7)) # change this to the layers you would like to use if needed

submodule_dict, _ = load_autoencoders(
    model, 
    layers,
    os.path.join(oai_sae_path, "autoencoders/")
)

tokens = load_tokenized_data(model.tokenizer,dataset_split="train")
cache = FeatureCache(model, submodule_dict, minibatch_size=32)
print('Cache created, going to run. This may take a moment...')
cache.run(tokens, n_tokens=10_000_000)

save_dir = os.path.join(feature_path, "raw_features/")
# Save the first 1000 features per layer
for layer in layers:
    feature_range = torch.arange(0,1000)
    cache.save_selected_features(feature_range, layer, save_dir=save_dir)

print('Finished. Features have been cached to ' + save_dir + '. Thank you for your patience!')