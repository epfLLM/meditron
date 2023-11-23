<img width=50% src="figures/meditron_LOGO.png" alt="MediTron logo" title="Meditron-logo">

MediTron is a suite of open-source medical Large Language Models (LLMs).

We release MediTron-7B and MediTron-70B, which are adapted to the medical domain from Llama-2 through continued pretraining on a comprehensively curated medical corpus, including selected PubMed papers and abstracts, a new dataset of internationally-recognized medical guidelines, and a general domain corpus.

MediTron-70B, finetuned on relevant data, outperforms Llama-2-70B, GPT-3.5 and Flan-PaLM on multiple medical reasoning tasks.

## Model Details

- **Developed by:** [EPFL LLM Team](https://huggingface.co/epfl-llm)
- **Model type:** Causal decoder-only transformer language model
- **Language(s):** English (mainly)
- **License:** [LLAMA 2 COMMUNITY LICENSE AGREEMENT](https://huggingface.co/meta-llama/Llama-2-70b/raw/main/LICENSE.txt)
- **Continue-pretrained from model:** [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b)
- **Context length:**  4k tokens
- **Input:**  Text only data
- **Output:**  Model generates text only
- **Status:** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we enhance model's performance.
- **Knowledge Cutoff:** August 2023
- **Trainer:** [epflLLM/Megatron-LLM](https://github.com/epfLLM/Megatron-LLM)
- **Paper:** *[MediTron-70B: Scaling Medical Pretraining for Large Language Models]()* **[ADD LINK]**

## How to use

You can load MediTron model directly from the [HuggingFace model hub](https://huggingface.co/epfl-llm/meditron-70B) as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70B")
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-70B")

# Add your custom code for inference here
```

## Medical Training Data

We release code to download and pre-process the data used to train MediTron.

MediTron’s domain-adaptive pre-training corpus *GAP-Replay* combines 48.1B tokens from four corpora:

- **Clinical <u>G</u>uidelines**: a new corpus of 46K clinical practice guidelines from various healthcare-related sources, including hospitals and international organizations,
- **Paper <u>A</u>bstracts**: 16.1M abstracts extracted from closed-access PubMed and PubMed Central papers,
- **Medical <u>P</u>apers**: full-text articles extracted from 5M publicly available PubMed and PubMed Central papers.
- **<u>Replay</u> dataset**: 400M tokens of general domain pretraining data sampled from [RedPajama-v1](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).

### Download instructions

You can download and pre-process the entire GAP-Replay corpus by running `./download.sh` in the `gap-replay` folder.

You can download 36K open-access articles from our *Guidelines* corpus from the [HuggingFace datasets hub](https://huggingface.co/datasets/epfl-llm/guidelines).

```python
from datasets import load_dataset

dataset = load_dataset("epfl-llm/guidelines")
```

You can scrape and clean all 46K guidelines (including closed-access sources) by running `./download.sh` in the `guidelines` folder.

More details can be found in the [GAP-Replay documentation](gap-replay/README.md).

## Uses

MediTron-70B is being made available for further testing and assessment as an AI assistant to enhance clinical decision-making and democratize access to an LLM for healthcare use. Potential use cases may include but are not limited to:

- Medical exam question answering
- Supporting differential diagnosis
- Disease information (symptoms, cause, treatment) query
- General health information query

It is possible to use this model to generate text, which is useful for experimentation and understanding its capabilities. It should not be used directly for production or work that may impact people.

We do not recommend using this model for natural language generation in a production environment, finetuned or otherwise.

### Downstream Use

Meditron-70B is a foundation model that can be finetuned, instruction-tuned, or RLHF-tuned for specific downstream tasks and applications.
The main way we have used this model is finetuning for downstream question-answering tasks, but we encourage using this model for additional applications.

Specific formatting needs to be followed to prompt our finetuned models, including the  `<|im_start|>`, `<|im_end|>` tags, and  `system`, `question`, `answer`  identifiers.

```python
"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>question
{prompt}<|im_end|>
<|im_start|>answer
"""
```

**Note**: the above formatting is not a requirement if you use your own formatting option for the finetuning of the model.

## Medical Benchmark Inference & Evaluation

### Requriments

Before you start, please install the necessary packages:

    vllm >= 0.2.1
    transformers >= 4.34.0
    datasets >= 2.14.6
    torch >= 2.0.1

For detailed instructions to run inference and evaluation with medical benchmarks, please read the documentations here [inference & evaluation instructions](./evaluation/README.md).

## Model Deployment

For detailed instructions to deploy meditron models and have a interactive chat session, please read the documentations here [Model Deployment](./deployment/README.md)

## Training Procedure

We used the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) distributed training library, a derivative of Nvidia's Megatron LM project, to optimize training efficiency.
Hardware consists of 16 nodes of 8x NVIDIA A100 (80GB) SXM GPUs connected by NVLink and NVSwitch with a single Nvidia ConnectX-6 DX network card and equipped with 2 x AMD EPYC 7543 32-Core Processors and 512 GB of RAM.
The nodes are connected via RDMA over Converged Ethernet.

Our three way parallelism scheme uses:

- Data Parallelism (DP -- different GPUs process different subsets of the batches) of 2,
- Pipeline Parallelism (PP -- different GPUs process different layers) of 8,
- Tensor Parallelism (TP -- different GPUs process different subtensors for matrix multiplication) of 8.

![Pipeline](figures/meditron-pipeline.png "Pipeline")
