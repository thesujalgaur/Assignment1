import os
import openai
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your CSV dataset
df = pd.read_csv("data/LoanData.csv")

# Convert DataFrame to a string description for context
def describe_data():
    return df.describe(include='all').to_string()

data_summary = describe_data()

# Function to ask OpenAI a question based on data
def ask_bot(question):
    prompt = f"""You are a data analyst. Use the following dataset summary and answer the question accurately.

DATA SUMMARY:
{data_summary}

QUESTION:
{question}

ANSWER:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=ask_bot,
    inputs=gr.Textbox(label="Ask a question about the loan data"),
    outputs="text",
    title="Loan Data Chatbot",
    description="Ask questions like 'What's the average loan amount?' or 'How many loans were approved?'"
)

iface.launch()