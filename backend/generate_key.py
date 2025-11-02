import os
import json
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Create the service account dictionary
service_account_data = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),  # Convert escaped newlines
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_CERT_URL"),
    "universe_domain": os.getenv("UNIVERSE_DOMAIN"),
}

# Write data to gee-service-key.json
with open("gee-service-key.json", "w") as f:
    json.dump(service_account_data, f, indent=2)

print("✅ gee-service-key.json created successfully!")
