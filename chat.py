import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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
        # Compare the correct_response with existing responses
        if any(correct_response.lower() == resp.lower() for resp in intent['responses']):
            intent['patterns'].append(sentence)
            break
    else:
        # If the response does not exist, create a new tag
        new_intent = {
            "tag": correct_response[:50].replace(' ', '_'),  # Create a tag from the response (limit to 50 chars)
            "patterns": [sentence],
            "responses": [correct_response]
        }
        intents['intents'].append(new_intent)

    # Save the updated intents
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)

while True:
    sentence = input('Tu: ')
    if sentence.lower() == "salir":
        break

    sentence_lower = sentence.lower()  # Convert input to lowercase
    sentence_tokens = tokenize(sentence_lower)  # Tokenize the input
    X = bag_of_words(sentence_tokens, all_words)  # Convert tokens to bag of words
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

    # Ask for feedback
    feedback = input(f"{bot_name}: Esta es la respuesta que esperabas? (si/no): ")
    if feedback.lower() == "no":
        correct_response = input("Por favor adjunta la respuesta correcta: ")
        add_feedback_to_intents(sentence_lower, correct_response)
