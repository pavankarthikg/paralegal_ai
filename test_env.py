import os
from dotenv import load_dotenv

load_dotenv()
print("Your key is:", os.getenv("OPENAI_API_KEY"))
