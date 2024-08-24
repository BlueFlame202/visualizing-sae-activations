"""
generate_act_dists_data.py

This script can be run to generate HTML pages displaying activation distributions along with examples showing tokens with the highest activations for each feature. One can generate other distributions and visualize other examples by modifying this script slightly.
"""

import sys
sys.path.append('../') # to allow for better organization

from deps.sae_auto_interp.utils import load_tokenized_data
from deps.sae_auto_interp.features import FeatureRecord

from settings import settings

from nnsight import LanguageModel
import numpy as np
import random

from io import BytesIO
import base64
import os

import matplotlib.pyplot as plt
from IPython.display import HTML

def color_text(text, intensity=0.5):
    intensity = max(0.0, min(1.0, intensity))
    color_value = int(255 * intensity)
    color_hex = f"#595388{color_value:02x}"
    return f'<span style="background-color: {color_hex};">{text}</span>'

def format_tokens_with_activations(model, tokens, activations):
    tokenizer = model.tokenizer

    html_content = f""
    for i in range(len(tokens)):
        piece = tokenizer.convert_ids_to_tokens(int(tokens[i]))
        piece = color_text(piece, activations[i]/10)
        html_content += piece
    return html_content.replace('Ġ', ' ')

def display_tokens_with_activations(model, tokens, activations):
    display(HTML(format_tokens_with_activations(model, tokens, activations)))

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Save with tight bounding box
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

def get_dist_and_feature_acts(model, feature_number, layers):
    records = []
    for layer in layers:
        module_name = f"layer{layer}"

        records.append(FeatureRecord.from_tensor(
            tokens,
            module_name,
            selected_features=[feature_number],
            raw_dir = raw_features_path,
            min_examples=200,
            max_examples=10000
        ))

        for record in records[-1]:

            examples = record.examples
            for example in examples:
                normalized_activations = (example.activations / record.max_activation)*10
                example.normalized_activations = normalized_activations.round()
            # top_200 = examples[:200] # you can explore with these!
            # random_20 = random.sample(examples, 20)
            # random_top_200_20 = random.sample(top_200,20)
            # split_top_200_20_all_20 = random_top_200_20 + random_20

    dist = []
    for record in records[1]:
        examples = record.examples
        for example in examples:
            normalized_activations = (example.activations / record.max_activation)*10
            normalized_activations = normalized_activations[normalized_activations != 0]
            dist.extend(normalized_activations)
        break 

    counts, bin_edges = np.histogram(dist, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, counts, label='Smoothed Histogram')
    plt.xlabel("Activation")
    plt.ylabel("Frequency")
    plt.title("Activations for Feature " + str(feature_number))

    # Adjust the layout to make room for the text
    counter = 0

    res = f''
    for record in records[1]:
        examples = record.examples
        for example in examples:
            if counter > 15: 
                break
            counter += 1
            example_tokens = example.tokens
            activations = example.normalized_activations
            res += format_tokens_with_activations(model, example_tokens.numpy(), activations.numpy()) + "<br />"

    img_base64 = plot_to_base64()
    return img_base64, res

if __name__ == "__main__":
  # general variables
  layers = range(0, 7, 2) # just even layers, choose which layers you want

  model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
  tokens = load_tokenized_data(model.tokenizer, dataset_repo="kh4dien/fineweb-100m-sample", dataset_split="train[:15%]")

  raw_features_path = os.path.join(settings["feature_dir"], "raw_features/")
  processed_features_path = os.path.join(settings["feature_dir"], "processed_features/")
  random.seed(22)

  # HTML content with placeholders for plot images and text
  html_template = """
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
          }}
          .content {{
            margin: 20px;
            border: 1px solid black;
            padding: 3%;
          }}
          img {{
            max-width: 100%;
            height: auto;
          }}
        </style>
      </head>
      <body>
        <div class="content" id="feature_{plot_num}">
          <h1>Feature {plot_num}</h1>
          <h2>Distribution of Activations</h2>
          <img src="data:image/png;base64,{plot_data}" alt="Plot {plot_num}">
          <div>
            <h2>Activating Tokens in Examples</h2>
            {text}
          </div>
        </div>
      </body>
    </html>
    """

  mod_value = 2

  html_content = ""
  for i in range(1000):
    plot_data, text = get_dist_and_feature_acts(model, i, layers)
    html_content += html_template.format(plot_num=i, plot_data=plot_data, text=text)
    plt.close()

    if i % mod_value == 0:
      print("Completed " + str(i) + " features.")

  out_path = os.path.join(settings["visuals_dir"], "output.html")
  with open(out_path, 'w') as f:
    f.write(html_content)