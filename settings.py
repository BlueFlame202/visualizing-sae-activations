
import os

settings = {}

settings['feature_dir'] = '/Users/aathreyakadambi/Documents/research/EleutherAI/lm-interpretability/visualizing-sae-activations/res/features'
settings['model_dir'] = '/Users/aathreyakadambi/Documents/research/EleutherAI/lm-interpretability/visualizing-sae-activations/res/models'
settings['visuals_dir'] = '/Users/aathreyakadambi/Documents/research/EleutherAI/lm-interpretability/visualizing-sae-activations/res/visuals'

settings['oai_autoencoder_dir'] = os.path.join(settings['model_dir'], 'oai_autoencoders/')
settings['eai_autoencoder_dir'] = os.path.join(settings['model_dir'], 'eai_autoencoders/') # won't be used right now, but maybe in the future