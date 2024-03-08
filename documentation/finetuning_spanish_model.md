# Finetuning Spanish model task

**Steps for test QLora on Epfl hugginface library**

- Select and LLM Spanish model and check if possible use QLora training process with it.

- Study and learn the QLora concept and Epfl using

- Check if possible the Meditron finetunning pipeline adapt to use QLora and single GPU training process

- Execute a tiny test with all configurations

# Criteria to select a Spanish LLM model for pretraining or Finetuning

- Evaluate the LLM pretraining only with a spanish text or more spanish text than other.

- The training library used in Meditron can use with the selected LLM

- The LLM can be used for finetunning and will be not expensive in time and memory

**Obtenined hugginface models:**


    projecte-aina/aguila-7b (Falcon base)
    clibrain/Llama-2-7b-ft-instruct-es (Llama2 base)
    TheBloke/Barcenas-Mistral-7B-GGUF (Mistral base)
    clibrain/lince-zero (Llama2 base)
    clibrain/Llama-2-13b-ft-instruct-es (Llama2 base)
    google/gemma-7b-it (Gemminis base)
    allenai/OLMo-7B (Olmo base)
    clibrain/Llama-2-13b-ft-instruct-es-gptq-4bit (Llama2 base)
    clibrain/lince-mistral-7b-it-es (Misttral base)
    Kukedlc/Llama-7b-spanish (Llama2 base)
    google/gemma-7b (Gemminis base)
    allenai/OLMo-1B (Olmo base)

**Conclusions:**
- The most used model are 7B parameters size
- Considers the time and memory for pre training LLama 2, Gemma or another model if you have enough data as state-of-the-art
- Evaluate a finnetuning context first with the three LLM models in the above list
- Use a LLM for pretrainig model if we have enough data


 
 
 
 
 
 **Research Objetive**
 - Analyze all spanish LLM model in hugginface repository built with fundation model as Llama2 or another
 
 
 
 **Research Criteria:**
 
 - Similar Meditron fundation model, for reutilice all used requeriments
 - Another model with competitive criteria as Mistral, Gemminis
 Licences Open Source or similar for next deployment in several context
 - Was built with spanish corpus or part of them used spanish language
 




# Resources about Spanish LLM Model

- **Spanish Language Models Refrences to resources** https://github.com/PlanTL-GOB-ES/lm-spanish
  
- **Biomedical and clinical language model for Spanish** https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es

- **Biomedical language model for Spanish** https://huggingface.co/PlanTL-GOB-ES/bsc-bio-es

