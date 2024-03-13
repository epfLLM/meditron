# Spanish Medical LLM

## Similars Spanish LLMS

“Sequence-to-Sequence Spanish Pre-trained Language Models” 

**Pretrained data:**

- OSCAR 21.09 corpus, which includes a deduplicated Spanish dataset of approximately 160GB of text
- mC4-es corpus (Xue et al., 2021), extensive 500GB text
- Spanish Wikipedia dump ttext from diverse sources, 10GB
- The ones from BSC but are BERT style.

**Finetunning data:**

- MLQA (Lewis et al., 2019) and SQAC (Gutiérrez-Fandiño et al., 2021) datasets
for this evaluation.
- MLQA presents a collection of parallel multi-lingual articles extracted from  Wikipedia and offers a development set and test set professionally translated into Spanish.
- SQAC was created exclusively for Spanish evaluation and contains articles extracted purely from Spanish sources.

## Meditron

Meditron is an open-source suite of medical Large Language Models (LLMs). It includes Meditron-7B and Meditron-70B, both adapted to the medical domain through pretraining on a curated medical corpus. Meditron-70B, when finetuned on relevant data, surpasses Llama-2-70B, GPT-3.5, and Flan-PaLM in various medical reasoning tasks.

The code is designed to operate in a distributed environment, with certain sections serving as a starting point. The evaluation specifically focuses on English language capabilities.

# Evaluation

Fine-tuning a Large Language Model (LLM) in the medical context is a sophisticated task that requires careful consideration of model architecture, data, and ethical considerations. The approach you choose will depend on the specific tasks you wish to perform with your model, such as information extraction, patient triage, generating medical reports, or answering medical questions. Downstream evaluation metrics differ based on the transformer architecture.

### 1. Only Encoder (BERT-like Models)

### Use Cases:

- **Named Entity Recognition (NER):** Identifying medical terms, medication names, and other specific entities in text.
- **Sentiment Analysis**: Classification of text.

### Approach:

- **Data Preparation:** Collect a diverse dataset of medical texts.
- **Preprocessing:** Normalize the medical texts (e.g., lowercasing, removing special characters) and annotate your data for the specific task, if necessary.
- **Fine-Tuning:** Use a pre-trained BERT model and fine-tune it on your medical dataset. You may need to adjust the model's architecture slightly depending on your task, such as adding a classification layer for NER or classification tasks.

### 2. Decoder-Only (GPT-like Models)

### Use Cases:

- **Generating Medical Text:** Generating discharge summaries, patient instructions, or creating medical content.
- **Question Answering:** Providing answers to medical questions based on a large corpus of medical knowledge.
- **Dialogue Systems:** Powering conversational agents for patient engagement or support.

### Approach:

- **Data Preparation:** Assemble a large corpus of medical texts, including dialogues (if available), Q&A pairs, and general medical information.
- **Preprocessing:** Similar to the BERT approach but ensure the texts are suitable for generative tasks.
- **Fine-Tuning:** Use a pre-trained GPT model and fine-tune it on your dataset. You may experiment with different prompts and fine-tuning strategies to improve performance on generative tasks.

### 3. Encoder-Decoder (T5, BART-like Models)

### Use Cases:

- **Translation:** Translating medical documents between languages.
- **Summarization:** Generating concise summaries of lengthy medical texts or patient histories.
- **Question Answering:** Especially for complex queries that require understanding and synthesizing information from multiple sources.

### Approach:

- **Data Preparation:** Collect a dataset that suits your specific task, such as parallel corpora for translation or long texts with summaries.
- **Preprocessing:** Prepare and clean your data, ensuring that it is in a format suitable for both encoding and decoding tasks.
- **Fine-Tuning:** Use a pre-trained model like T5 or BART and fine-tune it on your specific dataset. Tailor the input and output formats to match your task, such as "translate English to French" for translation tasks.

### Ethical and Fair Approach Considerations:

- **Bias and Fairness:** Be aware of and actively mitigate biases in your dataset and model. This includes biases related to gender, ethnicity, and age.
- **Data Privacy:** Ensure that the data used for training and fine-tuning respects patient confidentiality and complies with regulations.
- **Model Transparency:** Document the data sources, model decisions, and any limitations of your model.

## Tools

### Medplexity

Medplexity is a framework designed to explore the capabilities of LLMs in the medical domain. We achieve this by providing interfaces and collections of common benchmarks, LLMs, and prompts.

Medplexity automatically create the prompts for evaluation in QA. The prompts are in English, and the samples use the OpenAI API. For the Spanish case, I believe we should create our own framework.

### Olmo

Used insise catwalk and tango.

[https://github.com/allenai/OLMo-Eval](https://github.com/allenai/OLMo-Eval)

### Cawalk

[https://github.com/allenai/catwalk](https://github.com/allenai/catwalk)

### Tango

AI2 Tango replaces messy directories and spreadsheets full of file versions by organizing experiments into discrete steps that can be cached and reused throughout the lifetime of a research project.

[https://github.com/allenai/tango](https://github.com/allenai/tango)

### Deep Eval

**DeepEval** is a simple-to-use, open-source LLM evaluation framework. It is similar to Pytest but specialized for unit testing LLM outputs. DeepEval incorporates the latest research to evaluate LLM outputs based on metrics such as hallucination, answer relevancy, RAGAS, etc., which uses LLMs and various other NLP models that runs **locally on your machine** for evaluation.

Can evaluate ***local deploy*** models with this metrics:

- **G-Eval**: General performance evaluation across multiple tasks or criteria.
- **Summarization**: Assessing the ability to create concise and relevant summaries.
- **Answer Relevancy**: Measuring the relevance and accuracy of model responses to prompts.
- **Faithfulness**: Ensuring output accuracy and fidelity to the source material or input data.
- **Contextual Recall**: Evaluating the model's use of relevant information from the given context.
- **Contextual Precision**: Assessing the specificity and accuracy of the model's output in relation to the task context.
- **RAGAS**: Evaluating retrieval-augmented generation models for effective information use and text generation.
- **Hallucination**: Identifying instances where the model generates unsupported or false information.
- **Toxicity**: Measuring the propensity to produce harmful or inappropriate content.
- **Bias**: Assessing the presence of unfair prejudices or stereotypes in model outputs.

[https://github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)

## Based on task

Language is very complex, and the best evaluation is always a manual human one. However, other options exist:

**Question Answering (QA):**

- **Accuracy:** The percentage of questions answered correctly. This is a good starting point, but doesn't capture nuances.
- **F1 Score:** Combines precision (ratio of correct answers to retrieved answers) and recall (ratio of correct answers to all possible answers) into a single metric.
- **Mean Reciprocal Rank (MRR):** The average inverse of the rank of the first correct answer.

**Sentiment Analysis:**

- **Accuracy:** The percentage of correctly classified sentiment (positive, negative, or neutral).
- **Precision, Recall, and F1 Score:** Similar to QA, but for each sentiment class.
- **Error Analysis:** Examining misclassified examples to identify areas for improvement.

**Text Summarization:**

- **ROUGE Score:** Measures overlap in n-grams (sequences of n words) between the generated summary and reference summaries.
- **BLEU Score:** Similar to ROUGE, but with additional factors like brevity penalty.

**Text Generation (Depends on the specific task):**

- **Grammatical Correctness and Fluency:** How well-formed and natural the generated text reads.
- **Creativity and Coherence:** Does the generated text make sense and flow logically?

## Benchmarks

Each of these tasks and datasets targets specific capabilities of language models, from understanding and generating natural language to performing specialized reasoning and knowledge application across various domains, including general knowledge, science, mathematics, and ethics.

### General Tasks and Datasets

- **WikiText**: A dataset for language modeling tasks, consisting of articles from Wikipedia. It's used for training models on a wide range of topics for better text generation and understanding.
- **PIQA (Physical Interaction Question Answering)**: Focuses on reasoning about physical interactions with objects, requiring models to predict the outcome of physical actions in a given scenario.
- **SQuAD (Stanford Question Answering Dataset)**: A benchmark dataset for machine reading comprehension, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.

### SQuAD Shifts

These are variants of the original SQuAD (Stanford Question Answering Dataset) dataset adapted to different domains to test a model's generalization abilities:

- **SQuADShifts-Reddit, Amazon, NYT, New-Wiki**: Adaptations of the SQuAD dataset using content from Reddit, Amazon reviews, The New York Times articles, and newer Wikipedia articles, respectively.

### MRQA (Machine Reading for Question Answering) Datasets

A collection of datasets compiled for the MRQA shared task, aimed at evaluating models across a diverse set of reading comprehension tasks:

- **RACE, NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions, BioASQ, DROP, RelationExtraction, TextbookQA, Duorc.ParaphraseRC**: Each focuses on different aspects of reading comprehension, such as multiple-choice questions (RACE), news articles (NewsQA), trivia knowledge (TriviaQA), web search results (SearchQA), multi-hop reasoning (HotpotQA), real user queries (NaturalQuestions), biomedical questions (BioASQ), discrete reasoning over paragraphs (DROP), etc.

### Other Tasks and Datasets

- **SQuAD2**: An extension of SQuAD that includes unanswerable questions, making the task more challenging.
- **RTE (Recognizing Textual Entailment)**: Involves determining whether a given text logically follows from another text.
- **SuperGLUE::RTE, CoLA, MNLI, MRPC, QNLI, QQP, SST, WNLI, BoolQ, etc.**: Part of the SuperGLUE benchmark, these tasks involve various aspects of language understanding, such as entailment, grammaticality, natural language inference, paraphrase detection, question answering, and sentiment analysis.
- **LAMBADA**: A dataset for evaluating the capabilities of models in predicting the final word of a text passage, designed to test the understanding of context.
- **PubMedQA**: A dataset for biomedical question answering.
- **SciQ**: Focused on science exam questions.
- **QA4MRE**: A series of tasks from the Question Answering for Machine Reading Evaluation challenge, spanning several years.
- **ANLI (Adversarial NLI)**: A series of datasets for testing natural language understanding in an adversarial setting.
- **Ethics**: Tasks related to evaluating models on ethical reasoning, including deontology, justice, utilitarianism, and virtue ethics.
- **MathQA, Arithmetic, Anagrams, etc.**: Datasets focusing on mathematical reasoning, arithmetic operations, and word manipulation tasks.