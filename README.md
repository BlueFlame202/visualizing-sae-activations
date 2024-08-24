# Visualizing SAE Activations

This repository walks through getting started with working with some of the LLM interpretability research done in https://github.com/EleutherAI/sae-auto-interp/tree/Experiments. For now, the `sae_auto_interp` folder from that repository has been copied over here for simplicity, and a file in scripts includes much code from a relevant script from that repository. The variable `DEVICE` from various `__init__.py` files has also been set to `"cpu"`, but it can be changed to `cuda` if desired. 

Steps to use:
1. `git clone` this repository locally.
2. Edit the `settings.py` file to use the appropriate filepaths on your computer. It is better to use hard-coded file paths here rather than relative ones.
3. Run `conda env create -f environment.yml` to create the `lm-sae-interp-env` environment. Then run `conda activate lm-sae-interp-env`.
4. Run the `download_oai_sae.py` script to download the OpenAI SAE that you want. For example, this might look like:
`python -m scripts.download_oai_sae -1`.
5. Run the `cache_features.py` script to cache features. You should look through that file and modify it to use the layers, features, and model you want. You can then run it as `python -m scripts.cache_features`.
6. Run the `generate_act_dists_data.py`