import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to generate a response
def generate_response(user_input):
    # Tokenize the input
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Generate text using the model with attention mask and padding
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,  # or any other appropriate pad token ID
        attention_mask=None  # as attention mask is handled by the tokenizer
    )

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Streamlit UI
st.title("Conversation Model")

# User input text box
user_input = st.text_input("Your message:", "")


hide = """
<style>
div[data-testid="stConnectionStatus"] {
    display: none !important;
</style>
"""

st.markdown(hide, unsafe_allow_html=True)

# Generate button
if st.button("Generate Response"):
    # Generate response
    response = generate_response(user_input)

    # Display generated response
    st.text("Model Response: " + response)
