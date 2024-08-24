"""
download_oai_sae.py

This script is based on code from https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/paths.py. It makes it slightly easier to download some of the openai sparse autoencoders.
"""

import sys
sys.path.append('../') # to allow for better organization

from settings import settings

import argparse
import os
import blobfile as bf

# following two methods based on https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/paths.py

def get_blob_path_end(location, layer_index, version):
    assert layer_index in range(12)
    assert version in ["v1", "v4", "v5_32k", "v5_128k"]

    if version == 'v1':
        """
        Details:
        - Number of autoencoder latents: 32768
        - Number of training tokens: ~64M
        - Activation function: ReLU
        - L1 regularization strength: 0.01
        - Layer normed inputs: false
        - NeuronRecord files:
            `az://openaipublic/sparse-autoencoder/gpt2-small/{location}/collated_activations/{layer_index}/{latent_index}.json`
        """
        assert location in ["mlp_post_act", "resid_delta_mlp"]
        return f"gpt2-small/{location}/autoencoders/{layer_index}.pt"
    elif version == 'v4':
        """
        Details:
        same as v1
        """
        assert location in ["mlp_post_act", "resid_delta_mlp"]
        return f"gpt2-small/{location}_v4/autoencoders/{layer_index}.pt"
    elif version == 'v5_32k':
        """
        Details:
        - Number of autoencoder latents: 2**15 = 32768
        - Number of training tokens:  TODO
        - Activation function: TopK(32)
        - L1 regularization strength: n/a
        - Layer normed inputs: true
        """
        assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]
        return f"gpt2-small/{location}_v5_32k/autoencoders/{layer_index}.pt"
    elif version == 'v5_128k':
        """
        Details:
        - Number of autoencoder latents: 2**17 = 131072
        - Number of training tokens: TODO
        - Activation function: TopK(32)
        - L1 regularization strength: n/a
        - Layer normed inputs: true
        """
        assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]
        return f"gpt2-small/{location}_v5_128k/autoencoders/{layer_index}.pt"
        

def get_blob_path(location, layer_index, version):
    return f"az://openaipublic/sparse-autoencoder/" + get_blob_path_end(location, layer_index, version)

# NOTE from openai: we have larger autoencoders (up to 8M, with varying n and k) trained on layer 8 resid_post_mlp
# we may release them in the future

def download_oai_sae(location, layer_index, version):
    blob_path_end = get_blob_path_end(location, layer_index, version)
    blob_path = get_blob_path(location, layer_index, version)
    local_path = os.path.join(settings["oai_autoencoder_dir"], '/'.join(blob_path_end.split('/')[:-1]))
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    local_path = os.path.join(local_path, blob_path.split('/')[-1])

    with bf.BlobFile(blob_path, mode="rb") as in_file, open(local_path, "wb") as out_file:
        out_file.write(in_file.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and run sparse autoencoder")
    parser.add_argument('layer_index', type=int, help="The layer index (e.g., 6) or -1 to download all 12")
    parser.add_argument('--location', type=str, default='resid_post_mlp', help="The location (e.g., 'resid_post_mlp')")
    parser.add_argument('--version', type=str, default='v5_32k', help="The version (e.g., 'v5_32k')")

    args = parser.parse_args()
    
    print("Downloading. This may take a moment...")
    if args.layer_index != -1:
        download_oai_sae(args.location, args.layer_index, args.version)
    else:
        for i in range(12):
            download_oai_sae(args.location, i, args.version)
            print("Downloaded layer " + str(i) + "...")
    
    print("Successfully downloaded! Thank you for your patience!")