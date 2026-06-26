import os
import sys
from google import genai
from google.genai import types

# Define the Poke-inspired system prompt
POKE_SYSTEM_PROMPT = """
You are Poke, a snappy, witty, and highly proactive personal AI assistant. 
Your style is opinionated, conversational, and direct—think of yourself as a smart friend who gets things done, not a servile corporate chatbot.

Rules:
1. TONE: Snappy, slightly snarky but charming, and extremely concise. Use conversational shorthand where natural (e.g. "idk", "ykwim", "tbh").
2. PROACTIVE: Don't just answer questions; nudge the user to take action, close loops, or schedule tasks. 
3. ASSUME CAPABILITY: Always act as though you have access to the user's tools (calendar, reminders, email). 
4. SAFETY: For any action that modifies the user's state (like sending an email or deleting a reminder), draft it first and request confirmation.
5. TOOL CALL FORMAT: When you decide to call a tool, append a structured tool instruction block at the end of your response like this:
   [ACTION: CREATE_REMINDER | title: "..." | due: "..."]
"""

def main():
    # Make sure API key is set
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    # Initialize the new Google GenAI Client
    client = genai.Client()

    # Define user input (simulating a request to schedule something)
    user_message = "remind me to check on the pizza in 20 minutes and also tell me something funny"

    print(f"User: {user_message}\n")
    print("Sending request to Gemma 4 via Google AI Studio...")

    try:
        # Generate content using a Gemma 4 model (gemma-4-31b-it)
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=POKE_SYSTEM_PROMPT,
                temperature=0.7,
                # Optionally set thinking budget if supported by the model
                # thinking_config=types.ThinkingConfig(thinking_budget=1024)
            )
        )

        print("\nPoke's Response:")
        print(response.text)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nNote: If 'gemma-4-31b-it' is not yet enabled in your region/tier, you can try fallback models like 'gemma-4-26b-a4b-it' or 'gemma-2-27b-it'.")

if __name__ == "__main__":
    main()
