import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()                                    # reads .env → loads GROQ_API_KEY
client = Groq(api_key=os.getenv("GROQ_API_KEY")) # the key, pulled from env (never hardcoded)

def generate(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


