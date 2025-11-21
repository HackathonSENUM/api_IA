import os
import google.generativeai as genai

# Assure-toi que ta clé API est bien définie
genai.configure(api_key="AIzaSyCOTvmzgg38ok9Yib5-B-7KpfkU9OzVDvA")

# Liste tous les modèles disponibles
print("Modèles disponibles pour ton projet :")
for model in genai.list_models():
    print(f"- {model.name}")
