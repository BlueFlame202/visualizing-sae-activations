# from .EleutherAI import load_eai_autoencoders # commented out for now because it was causing issues with directory structure TODO: change later when updating to latest version of the repository
from .OpenAI import load_oai_autoencoders
from .Sam import load_sam_autoencoders

def load_autoencoders(model, ae_layers, weight_dir, **kwargs):

    if "gpt2" in weight_dir: # edited this a bit
        submodules,model = load_oai_autoencoders(model, ae_layers, weight_dir)
       
    # if "llama" in weight_dir:
    #     submodules,model = load_eai_autoencoders(model, ae_layers, weight_dir)
    # if "nora" in weight_dir:
    #     submodules,model = load_eai_autoencoders(model, ae_layers, weight_dir)

    if "pythia" in weight_dir:
        submodules,model = load_sam_autoencoders(model, ae_layers, weight_dir, **kwargs)
    
    return submodules,model
