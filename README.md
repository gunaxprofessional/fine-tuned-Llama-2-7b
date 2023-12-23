# Fine Tuned Llama 2 7b on Html Data

## File 1: dataset_creation.py

**Summary:**
The `dataset_creation` file encompasses the entire process of creating a dataset for the project. It covers library installation, dataset loading, tokenization, near-deduplication, data analysis, prompt template transformation, dataset export, data splitting, and pushing the dataset to the Hugging Face Hub.

## File 2: fine_tune_llm.py

**Summary:**
The `fine_tune_llm` file focuses on fine-tuning the Llama-2 language model using the HTML dataset. It includes setup and imports, model and dataset configuration, BitsAndBytesConfig and device configuration, loading and configuring the model, text generation pipeline, model architecture explanation, PEFT configuration, supervised fine-tuning trainer, and save and cleanup operations.

## File 3: run_fine_tuned_llama.py

**Summary:**
In `run_fine_tuned_llama`, the pre-trained Llama-2 model is initialized and configured for quantization. The code loads the model, sets up a chatbot pipeline, and executes it to generate HTML code based on a predefined question. The focus is on utilizing the fine-tuned model for practical HTML code generation.

## File 4: streamlit_app.py

**Summary:**
The `streamlit_app` file introduces a Streamlit application for the project. It installs necessary packages, creates a Streamlit app generating HTML code based on user instructions, and displays a conversation history. The app is run in the background, and a publicly accessible URL is provided through localtunnel for users to access the Streamlit app.

## Evaluation
To evaluate the fine-tuned LLama, I am currently exploring various metrics such as ROUGE, BLEU and so on. However, I have not yet found a metric specifically designed for evaluating the fine-tuned LLama on HTML data. I hope to discover a suitable method for evaluating the LLama, with a particular emphasis on its performance in generating HTML.
