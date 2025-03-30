import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from models.sentimenent_analysis import analyze_sentiment
from models.context_retreival import retrieve_relevant_context, get_stress_related_advice, get_resource_suggestions
from intent_matcher import match_intent  

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def generate_response(user_input):
    """Generates chatbot response using Gemini API or intent fallback."""

    # Step 1: Try to match an intent from KB
    tag, matched_response = match_intent(user_input)
    if matched_response:
        return matched_response

    # Step 2: Use Gemini with context if no match
    retrieved_context = retrieve_relevant_context(user_input)
    sentiment = analyze_sentiment(user_input)
    stress_advice = get_stress_related_advice()
    resource_suggestions = get_resource_suggestions()

    if retrieved_context == "No relevant context found.":
        retrieved_context = ""

    prompt = f"""
    You are a mental health chatbot offering **concise and relevant** responses.
    Always prioritize **user input** over retrieved context.
    
    User Input: "{user_input}"
    Detected Sentiment: {sentiment}

    **Relevant Context (if applicable)**:
    {retrieved_context}

    **Brief Advice**:
    {stress_advice}

    **Resource Suggestions (if needed)**:
    {resource_suggestions}

    💡 **Response Guidelines**:
    - Keep answers **brief (max 3-4 sentences)**
    - If retrieved context is irrelevant, **ignore it**
    - If unsure, **ask a follow-up question**
    """

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
    except Exception:
        return "Sorry, I am experiencing technical issues at the moment."


if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye! Take care. 😊")
            break
        bot_response = generate_response(user_input)
        print(f"\nBot: {bot_response}")




# import google.generativeai as genai
# from config import GEMINI_API_KEY, GEMINI_MODEL
# from models.sentimenent_analysis import analyze_sentiment
# from models.context_retreival import retrieve_relevant_context, get_stress_related_advice, get_resource_suggestions

# # Configure Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

# def generate_response(user_input):
#     """Generates chatbot response using Google Gemini API."""

#     retrieved_context = retrieve_relevant_context(user_input)
#     sentiment = analyze_sentiment(user_input)
#     stress_advice = get_stress_related_advice()
#     resource_suggestions = get_resource_suggestions()

#     print("📌 Building structured prompt for Gemini API...")
    
#     # Ensure retrieved context is actually relevant
#     if retrieved_context == "No relevant context found.":
#         retrieved_context = ""

#     prompt = f"""
#     You are a mental health chatbot offering **concise and relevant** responses.
#     Always prioritize **user input** over retrieved context.
    
#     User Input: "{user_input}"
#     Detected Sentiment: {sentiment}

#     **Relevant Context (if applicable)**:
#     {retrieved_context}

#     **Brief Advice**:
#     {stress_advice}

#     **Resource Suggestions (if needed)**:
#     {resource_suggestions}

#     💡 **Response Guidelines**:
#     - Keep answers **brief (max 3-4 sentences)**
#     - If retrieved context is irrelevant, **ignore it**
#     - If unsure, **ask a follow-up question**
#     """

#     print("📌 Sending request to Google Gemini API...")
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         response = model.generate_content(prompt)

#         return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
    
#     except Exception as e:
#         return "Sorry, I am experiencing technical issues at the moment."

# # Run chatbot in CLI mode
# if __name__ == "__main__":
#     while True:
#         user_input = input("\nUser: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("\n👋 Goodbye! Take care. 😊")
#             break
        
#         bot_response = generate_response(user_input)
