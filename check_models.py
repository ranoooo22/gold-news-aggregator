import os
from google import genai

# Holt den API-Key aus deinen GitHub Secrets
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("Fehler: Kein API-Key gefunden!")
else:
    client = genai.Client(api_key=API_KEY)
    print("Folgende Modelle sind für deinen Key freigeschaltet:")
    print("-" * 50)
    
    # Ruft die Liste aller verfügbaren Modelle ab
    for model in client.models.list():
        print(model.name)
