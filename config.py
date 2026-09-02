"""
Configuration module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for loading configuration values (like API keys) from
environment variables. No secrets are hardcoded here, and no API calls
are made in this stage — this just prepares the settings for later use.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file into the process environment.
# In production, these would instead be set as real environment variables.
load_dotenv()

class Config:
    """Holds application configuration loaded from environment variables."""

    # Gemini API key — will be used once the AI features are implemented.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    @classmethod
    def validate(cls):
         """
        Basic sanity check that required configuration is present.
 
        Stage 1 doesn't call the Gemini API yet, so this just warns
        instead of raising an error.
        """
         if not cls.GEMINI_API_KEY:
             print("Warning: GEMINI_API_KEY is not set. Add it to your .env file.")


# A ready-to-use config instance other modules can import.
config = Config()