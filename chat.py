import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from openai import OpenAI
import pandas as pd
import os

api_key = ""  
client = OpenAI(api_key=api_key)

# Load the CSV file (using an absolute or relative path)
csv_file_path = '/Users/santiagodegrandchant/Desktop/msc_banco/netflix_userbase.csv'

# Load the data into a pandas DataFrame
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Automatically generate the schema from the DataFrame
schema = {
    "table": "Sheet 1",  # Replace with the appropriate table name
    "columns": df.columns.tolist()
}

# Function to generate sample values
def generate_sample_values(df, num_samples=5):
    sample_values = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        sample_values[column] = unique_values[:num_samples].tolist()
    return sample_values

# Generate sample values
sample_values = generate_sample_values(df)

# Print schema for verification
print("Schema:", schema)
print("Sample Values:", sample_values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Marce"
print("Chateemos! Escribe 'salir' para salir del chat")

def add_feedback_to_intents(sentence, correct_response):
    # Load existing intents
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    
    # Check if the correct_response already exists in intents
    for intent in intents['intents']:
        if any(correct_response.lower() == resp.lower() for resp in intent['responses']):
            intent['patterns'].append(sentence)
            break
    else:
        # If the response does not exist, create a new tag
        new_intent = {
            "tag": correct_response[:50].replace(' ', '_'),
            "patterns": [sentence],
            "responses": [correct_response]
        }
        intents['intents'].append(new_intent)

    # Save the updated intents
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)

def get_sql_query_from_chatgpt(prompt, schema, sample_values):
    structured_prompt = f"""Dada la siguiente esquema de tabla y algunos valores de ejemplo:
Esquema de tabla:
{json.dumps(schema, indent=2)}

Valores de ejemplo:
{json.dumps(sample_values, indent=2)}

genera una condición de consulta Pandas de una sola línea para el siguiente pedido:

"{prompt}"

La condición debe ser válida para DataFrame.query() y no estar entrecomillada.
El resultado que me das solo puede contener la condición de consulta, no está permitido nada más.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un ingeniero de consultas pandas."},
                {"role": "user", "content": structured_prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # Extract the condition from the response and ensure it's a single line
        condition = response.choices[0].message.content.strip()
        return condition
    except Exception as e:
        return f"Error: {e}"

def query_csv(condition):
    try:
        # Use pandas query to filter the DataFrame
        result = df.query(condition)
        return result.to_string(index=False)
    except Exception as e:
        return f"Error: {e}"

while True:
    sentence = input('Tu: ')
    if sentence.lower() == "salir":
        break

    # Check for keyword to trigger SQL query generation
    if "sql" in sentence.lower():
        sql_condition = get_sql_query_from_chatgpt(sentence, schema, sample_values)
        print(f"{bot_name}: Generé el siguiente query: {sql_condition}")
        
        # Query the CSV file
        result = query_csv(sql_condition)
        print(f"{bot_name}: Resultado:\n '{result}'")
        continue

    sentence_lower = sentence.lower()
    sentence_tokens = tokenize(sentence_lower)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.70:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: No entiendo...")

    feedback = input(f"{bot_name}: Esta es la respuesta que esperabas? (si/no): ")
    if feedback.lower() == "no":
        correct_response = input("Por favor adjunta la respuesta correcta: ")
        add_feedback_to_intents(sentence_lower, correct_response)