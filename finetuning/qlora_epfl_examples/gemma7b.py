# -*- coding: utf-8 -*-
"""gemma7b.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1APcnivkPt6jdZzW2lJe6cHataQ4zBvqe

<!-- Banner Image -->
<img src="https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdevnotebooks.png" width="100%">

<!-- Links -->
<center>
  <a href="https://console.brev.dev" style="color: #06b6d4;">Console</a> •
  <a href="https://brev.dev" style="color: #06b6d4;">Docs</a> •
  <a href="/" style="color: #06b6d4;">Templates</a> •
  <a href="https://discord.gg/NVDyv7TUgJ" style="color: #06b6d4;">Discord</a>
</center>

# Run Google's Gemma🤙

Welcome!

In this notebook and tutorial, we will download & run Google's new [Gemma 7B-parameter model](https://huggingface.co/google/gemma-7b). From the Hugging Face page:
> Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

If you would like a fine-tuning guide, join our [Discord](https://discord.gg/y9428NwTh3) or message me on [X](https://x.com/harperscarroll) to let me know!

#### Before we begin: A note on OOM errors

If you get an error like this: `OutOfMemoryError: CUDA out of memory`, tweak your parameters to make the model less computationally intensive. I will help guide you through that in this guide, and if you have any additional questions you can reach out on the [Discord channel](https://discord.gg/y9428NwTh3) or on [X](https://x.com/harperscarroll).

To re-try after you tweak your parameters, open a Terminal ('Launcher' or '+' in the nav bar above -> Other -> Terminal) and run the command `nvidia-smi`. Then find the process ID `PID` under `Processes` and run the command `kill [PID]`. You will need to re-start your notebook from the beginning. (There may be a better way to do this... if so please do let me know!)

### Let's begin!

## 1. Get & Set Up a GPU

I used a GPU and dev environment from [brev.dev](https://brev.dev). Click the badge below to get your preconfigured instance:

[![](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/environment/new?instance=A10G:g5.xlarge&name=gemma&file=https://github.com/brevdev/notebooks/raw/main/gemma.ipynb&python=3.10&cuda=12.0.1)

Once you've checked out your machine and landed in your instance page, select the specs you'd like (I used Python 3.10 and CUDA 12.0.1; these should be preconfigured for you if you use the badge above) and click the "Build" button to build your verb container. Give this a few minutes.

A few minutes after your model has started Running, click the 'Notebook' button on the top right of your screen once it illuminates (you may need to refresh the screen). You will be taken to a Jupyter Lab environment, where you can upload this Notebook.

Note: You can connect your cloud credits (AWS or GCP) by clicking "Org: " on the top right, and in the panel that slides over, click "Connect AWS" or "Connect GCP" under "Connect your cloud" and follow the instructions linked to attach your credentials.

Now, run the cell below (Shift + Enter when you've highlighted the cell) to import the required libraries onto the GPU.
"""

!pip install -q -U torch torchvision torchaudio transformers jupyter ipywidgets accelerate

"""## 2. Download Gemma 7B

Now that we've installed the necessary libraries, let's pull the BioMistral model from Hugging Face.

First, request access to Gemma by clicking [here](https://huggingface.co/google/gemma-2b). Then, open a Terminal ('+' or 'Launcher' in the tabs above, then 'Terminal') and input `huggingface-cli login` and press Enter. Get a token by following the [link](https://huggingface.co/settings/tokens) (make sure you're logged in) and input it where you're prompted `Token:`. Note that for your safety/privacy, you won't see any text output when you paste the token into the Terminal shell.  

Check the output. If you see red text, you then may have to run `git config --global credential.helper store` and then `huggingface-cli login` again.

Once you see `Login successful`, you can run the cell below.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-7b"
# if you'd like, you can replace this with the instruct model (good for Q&A; like chatbot) by uncommenting the cell below
# model_id = "google/gemma-7b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=True,
    device_map="auto",
    trust_remote_code=True,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

eval_tokenizer = AutoTokenizer.from_pretrained(model_id,
                                               quantization_config=bnb_config,
                                               add_bos_token=True,
                                               trust_remote_code=True)

"""## 3. Run the Model!
Replace `eval_prompt` with your prompt string below, and run the model!
"""

eval_prompt = "The best way to "
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))

"""### Sweet... it worked!

Enjoy playing around with the full version of Gemma 2B!

One thing I want to share: note the difference between the standard and inference models!

Given the input `"The best way to "`, the outputs are:

**Standard:**
>The best way to <strong>get rid of a skunk</strong> is by using a <strong>skunk repellent</strong>.
>
>Skunks are known for their foul-smelling spray, which can be quite unpleasant if you come into contact with it.
>
>If you have a skunk problem in your yard or garden, there are several ways to get rid of them without harming them.
>
>In this article, we will discuss the different methods that you can use to get rid of skunks and how to prevent them from coming back.

**Instruct:**
>The best way to <b>find the perfect apartment in [city/neighborhood]</b> is to:
>
> **1. Define your needs and preferences:**
> * **Location:** Consider commute time, proximity to public transportation, and access to amenities like shopping, dining, and entertainment.
> * **Size and layout:** Think about how many bedrooms you need, the amount of living space, and whether you prefer a studio or one-bedroom apartment.
> * **Budget:** Determine your maximum monthly rent and be prepared to negotiate based

See how one is more generating one-way content, i.e. an article (standard) and the other generates more of a chatbot response (inference)?

You can also try prompting with full questions and see how both do. You'll find that the Instruct model is far more like a chatbot, providing you with useful answers right out of the box, without much prompt engineering.


Anyway, I hope you enjoyed this tutorial on running Gemma 2B. If you have any questions or suggestions, feel free to reach out to me on [X](https://x.com/harperscarroll) or on the [Discord](https://discord.gg/T9bUNqMS8d).

🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙
"""