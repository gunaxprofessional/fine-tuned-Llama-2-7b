import streamlit as st
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# Load the model and tokenizer
model_id = "Guna0pro/llama-2-7b-html"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Create a text-generation pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Create a prompt template
template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain
llm_chain = LLMChain(prompt=prompt, llm=local_llm)

# Streamlit app


def main():
    st.title("Fine-tuned LLM 2.7b on HTML data.")

    # Create a list to store the conversation history
    conversation_history = []

    # Get user input
    user_input = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        # Run the chatbot logic
        answer = llm_chain.run(user_input)

        # Display the answer
        st.markdown(f"**Answer:** {answer}")

        # Add the current question and answer to the conversation history
        conversation_history.append((user_input, answer))

    # Display the conversation history
    st.subheader("Conversation History")
    for question, answer in conversation_history:
        st.text(f"User: {question}")
        st.text(f"Bot: {answer}")
        st.text("-" * 30)


if __name__ == "__main__":
    main()
