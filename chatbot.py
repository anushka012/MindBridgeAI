# import openai
# # from config import GPT_MODEL
# from config import GPT_MODEL, OPENAI_API_KEY
# from models.sentiment_analysis import analyze_sentiment
# from models.context_retrieval import retrieve_relevant_context, get_stress_related_advice, get_resource_suggestions

# # Initialize OpenAI Client (NEW SYNTAX)
# # client = openai.OpenAI()
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

# def generate_response(user_input):
#     """Generates chatbot response using OpenAI's Chat API."""
    
#     print("📌 Retrieving relevant mental health context...")
#     retrieved_context = retrieve_relevant_context(user_input)  
#     print("✅ Relevant context retrieved.")

#     print("📌 Performing sentiment analysis...")
#     sentiment = analyze_sentiment(user_input)  
#     print(f"✅ Sentiment detected: {sentiment}")

#     print("📌 Getting stress-related advice...")
#     stress_advice = get_stress_related_advice()
#     print("✅ Stress-related advice retrieved.")

#     print("📌 Fetching mental health resource suggestions...")
#     resource_suggestions = get_resource_suggestions()
#     print("✅ Mental health resources retrieved.")

#     print("📌 Building structured prompt for OpenAI API...")
#     prompt = f"""
#     You are a supportive, fictional therapist chatbot.
#     Respond kindly and empathetically to the user's concerns.

#     User Input: "{user_input}"
#     Detected Sentiment: {sentiment}

#     Relevant Mental Health Context:
#     {retrieved_context}

#     Stress Advice:
#     {stress_advice}

#     Mental Health Resource Suggestions:
#     {resource_suggestions}

#     Your response:
#     """
#     print("✅ Prompt built successfully!")

#     print("📌 Sending request to OpenAI API...")
#     try:
#         response = client.chat.completions.create(  # ✅ NEW OpenAI API format
#             model=GPT_MODEL,
#             messages=[{"role": "system", "content": prompt}],
#             max_tokens=150
#         )
#         print("✅ OpenAI API responded successfully!")

#         return response.choices[0].message.content  # ✅ NEW way to access response
    
#     except openai.OpenAIError as e:  # ✅ Updated error handling
#         print(f"⚠️ OpenAI API Error: {str(e)}")
#         return "Sorry, I am experiencing technical issues at the moment."

#     except Exception as e:
#         print(f"⚠️ Unexpected Error: {str(e)}")
#         return "Something went wrong. Please try again."

# # Run chatbot in CLI mode
# if __name__ == "__main__":
#     print("\n🧠 Welcome to the fictional therapist chatbot!")
#     print("💬 Type your message and get support. Type 'exit' to quit.")

#     while True:
#         user_input = input("\nUser: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("\n👋 Goodbye! Take care. 😊")
#             break
        
#         print("\n🤖 Generating chatbot response...")
#         bot_response = generate_response(user_input)
#         print("\nBot:", bot_response)
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from models.sentimenent_analysis import analyze_sentiment
from models.context_retreival import retrieve_relevant_context, get_stress_related_advice, get_resource_suggestions

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def generate_response(user_input):
    """Generates chatbot response using Google Gemini API."""

    retrieved_context = retrieve_relevant_context(user_input)
    sentiment = analyze_sentiment(user_input)
    stress_advice = get_stress_related_advice()
    resource_suggestions = get_resource_suggestions()

    print("📌 Building structured prompt for Gemini API...")
    
    # Ensure retrieved context is actually relevant
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

    print("📌 Sending request to Google Gemini API...")
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)

        return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
    
    except Exception as e:
        return "Sorry, I am experiencing technical issues at the moment."

# def generate_response(user_input):
#     """Generates chatbot response using Google Gemini API."""
    
#     retrieved_context = retrieve_relevant_context(user_input)

#     sentiment = analyze_sentiment(user_input)

#     stress_advice = get_stress_related_advice()

#     resource_suggestions = get_resource_suggestions()

#     print("📌 Building structured prompt for Gemini API...")
#     prompt = f"""
#     You are a supportive, fictional therapist chatbot.
#     Respond kindly and empathetically to the user's concerns.

#     User Input: "{user_input}"
#     Detected Sentiment: {sentiment}

#     Relevant Mental Health Context:
#     {retrieved_context}

#     Stress Advice:
#     {stress_advice}

#     Mental Health Resource Suggestions:
#     {resource_suggestions}

#     Your response:
#     """

#     print("📌 Sending request to Google Gemini API...")
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         response = model.generate_content(prompt)

#         return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
    
#     except Exception as e:
#         return "Sorry, I am experiencing technical issues at the moment."

# Run chatbot in CLI mode
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye! Take care. 😊")
            break
        
        bot_response = generate_response(user_input)
