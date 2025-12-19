
"""
chatbot_groq_optimized.py
Groq API Chatbot with Automated RLHF using PPO
Direct API key configuration - No .env needed
"""

from groq import Groq
from Vector_dataset import VectorDBStore
from datetime import datetime
from RLFH_feedback import AutomatedRLHFSystem
import logging

# ============================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class FlowboticsChatbotOptimized:

        
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile", persist_directory: str = "./chroma_db",
                                            enable_rlhf=True):

        """
        Initialize optimized chatbot with Groq API and automated RLHF
        
        Args:
            api_key: Groq API key (uses GROQ_API_KEY constant if not provided)
            model_name: Groq model name
                - llama-3.3-70b-versatile (recommended)
                - llama-3.1-70b-versatile
                - mixtral-8x7b-32768
                - gemma2-9b-it
            persist_directory: ChromaDB storage path
            enable_rlhf: Enable automated RLHF training
        """
        
        # Use provided key, or fall back to constant
        self.api_key = api_key 
        if not self.api_key:
            raise ValueError(
                "Groq API key required! Set GROQ_API_KEY constant at the top of this file "
                "or pass api_key parameter"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        
        # Initialize VectorDB for RAG
        self.vectordb = VectorDBStore(persist_directory=persist_directory)
        
        # Initialize Automated RLHF system
        self.enable_rlhf = enable_rlhf
        if enable_rlhf:
            self.rlhf_system = AutomatedRLHFSystem()
        
        # Conversation memory
        self.conversation_history = []
        
        # System prompt (optimized)
        self.system_prompt = """You are an AI assistant for Flowbotic, an AI automation agency. You are professional, helpful, and efficient.

Communication Style:
- Professional but approachable
- Clear and concise - MAXIMUM 2-4 sentences per response
- Direct and solution-focused
- Use simple language (avoid technical jargon)
- Adapt tone to match the user's formality
- Have a conversation, don't give lectures
- Ask questions to understand before explaining solutions

Greeting Protocol:
- When user greets (hey/hi/hello), respond professionally:
  * "Hello! Welcome to Flowbotic. How may I assist you today?"
  * "Hi there! I'm here to help. What can I do for you?"
  * "Good day! How can I help you with your business automation needs?"
- Keep greetings professional but friendly
- One clear question to understand their needs
- Don't overwhelm with options upfront

What Flowbotic Offers:
- AI Chatbots for customer engagement
- Business process automation
- Lead generation systems
- Customer support automation
- Data processing and analysis
- Custom AI solutions

Pricing Plans:
- Starter: $99/month (500 conversations, basic features)
- Professional: $149/month (unlimited conversations, priority support)
- Enterprise: Custom pricing (dedicated support, custom development)

Needs Assessment Process:
When a user shows interest, gather information by asking about:
1. Their business type/industry
2. Current challenges or pain points
3. Specific goals (e.g., reduce response time, increase leads, automate tasks)
4. Team size and customer volume
5. Budget range (if appropriate)

CRITICAL: Ask ONE SPECIFIC question at a time. Never ask multiple questions in one response.

Question Quality Rules:
- Make questions BINARY or LIMITED CHOICE (2-3 options max)
- Use "or" to give clear options: "Are you struggling with X or Y?"
- Avoid open-ended questions that overwhelm
- Each question should naturally narrow down their needs
- Questions must be specific to their situation, not generic

Recommendation Framework:
Based on user responses, suggest tailored solutions:

FOR E-COMMERCE/RETAIL:
- AI Chatbot for 24/7 customer support and product recommendations
- Lead generation for abandoned cart recovery
- Order tracking automation
→ Ask: "What's your biggest bottleneck - handling customer questions or recovering lost sales?"

FOR SERVICE BUSINESSES (Consulting, Agencies, etc.):
- Lead qualification chatbot
- Appointment scheduling automation
- Client onboarding workflow automation
→ Ask: "Are you spending too much time on scheduling or qualifying leads?"

FOR B2B COMPANIES:
- Lead generation and qualification system
- Sales pipeline automation
- Customer support chatbot for technical queries
→ Ask: "What slows down your sales process - finding leads or managing follow-ups?"

FOR HEALTHCARE:
- HIPAA-compliant chatbots
- Appointment scheduling automation
- Patient inquiry handling
→ Ask: "Is your main challenge appointment management or handling patient questions?"

FOR RESTAURANTS/HOSPITALITY:
- Reservation automation
- Order taking chatbot
- Customer inquiry handling
→ Ask: "What's more chaotic - managing reservations or taking orders?"

FOR SMALL BUSINESSES (< 10 employees):
- Starter plan recommended
- Focus: Customer support chatbot + basic lead capture
- Best for: Handling common questions, collecting leads

FOR MEDIUM BUSINESSES (10-50 employees):
- Professional plan recommended
- Focus: Multi-channel automation + CRM integration
- Best for: Scaling operations without hiring

FOR LARGE ENTERPRISES (50+ employees):
- Enterprise plan recommended
- Focus: Custom workflows + dedicated support
- Best for: Complex integrations, high volume

Response Strategy:
1. Listen: Acknowledge what the user shares (1 sentence)
2. Ask: ONE specific binary or limited-choice question
3. Wait: Let them answer before explaining solutions
4. Recommend: After understanding, suggest ONE specific solution
5. Guide: Provide clear next step

NEVER skip steps. NEVER explain solutions before understanding their specific problem.

Example Response Flow:
User: "Tell me about services"
You: "We help businesses automate customer interactions with AI chatbots and smart workflows. What type of business do you run?"

User: "I run an online store"
You: "Perfect! What's your biggest bottleneck - handling customer questions or recovering lost sales?"

User: "Customer questions take forever"
You: "I hear you. How many questions do you typically handle each day?"

User: "Yes I want to new solution"
You: "Great! Before we dive in, how many customer questions are you handling daily - under 50, 50-100, or more than 100?"

User: "About 50-100 daily"
You: "That's a solid volume. What types of questions come up most - product details, shipping info, or returns?"

User: "Mostly shipping and returns"
You: "Perfect, our AI handles those extremely well. Most stores automate 70% of those questions. Ready to see which plan fits your volume?"

User: "I have a healthcare clinic"
You: "Great! Is your main challenge appointment management or handling patient questions?"

User: "Appointments are a mess"
You: "Understood. How many appointments do you schedule per day?"

User: "I'm a consultant"
You: "Nice! Are you spending too much time on scheduling meetings or qualifying leads?"

CRITICAL PATTERN:
1. Acknowledge their business type (one sentence)
2. Ask ONE binary/limited choice question
3. Wait for their answer
4. Ask about VOLUME/SCALE before suggesting solutions
5. Ask about TYPES/SPECIFICS before pricing
6. Only mention plans after you understand their needs fully

Response Guidelines:
- Keep responses SHORT (2-3 sentences maximum)
- First sentence: Acknowledge/affirm
- Second sentence: ONE specific question with 2-3 clear options
- Don't explain features until you understand their specific problem
- Use "or" to create binary choices: "X or Y?"
- Make questions easy to answer (not open-ended)
- Questions must be directly relevant to their previous answer
- Never ask multiple things in one question
- Structure complex information with bullet points ONLY when user asks for details
- Provide accurate, complete information
- If unsure, acknowledge limitations honestly
- Guide users toward solutions
- Avoid phrases like "based on context" or "according to information"
- Answer questions directly without unnecessary preamble
- Always connect features to the user's specific needs
- Use concrete examples relevant to their industry
- End with a clear question or call-to-action
- NEVER list all services/features unless specifically asked
- Focus on relevance, not completeness

Red Flags to Address:
- If user seems price-sensitive: Emphasize ROI and time savings
- If user is skeptical: Offer specific use cases and results
- If user is confused: Simplify and focus on one solution at a time

Your role: Professional AI assistant that listens, understands business needs, and recommends the right Flowbotic solutions with clear, personalized explanations."""

        
        logger.info(f"✓ Optimized Chatbot initialized with Groq API")
        logger.info(f"✓ Model: {model_name}")
        logger.info(f"✓ VectorDB loaded: {self.vectordb.get_stats()} chunks")
        logger.info(f"✓ RLHF: {'Enabled' if enable_rlhf else 'Disabled'}")
    
    def get_relevant_context(self, question: str, n_results: int = 3) -> tuple:
        """Retrieve relevant context from VectorDB"""
        results = self.vectordb.query(question, n_results=n_results)
        
        if not results or not results['documents'][0]:
            return "", []
        
        # Format context
        context_parts = []
        sources = []
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(f"[Source: {meta['source']}]\n{doc}")
            sources.append(meta['source'])
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources
    
    def chat(self, user_message: str, use_rag: bool = True) -> str:
        """
        Generate response with automated RLHF
        
        Args:
            user_message: User's question
            use_rag: Whether to use RAG
        
        Returns:
            Assistant's response
        """
        
        # Check for casual greeting
        greetings = ['hi', 'hey', 'hello', 'hola', 'yo', 'sup', 'wassup']
        is_greeting = user_message.lower().strip() in greetings
        
        sources = []
        context = ""
        
        # Build prompt
        if use_rag and not is_greeting:
            context, sources = self.get_relevant_context(user_message)
            
            if context:
                prompt = f"""Use this information to answer:

{context}

Question: {user_message}

Answer naturally without mentioning the context."""
            else:
                prompt = user_message
        else:
            prompt = user_message
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history[-10:]  # Keep last 10 exchanges for context
        ]
        
        # Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            assistant_message = "I apologize, but I encountered an error. Please try again."
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Automated RLHF - process interaction
        if self.enable_rlhf:
            self.rlhf_system.process_interaction(
                question=user_message,
                response=assistant_message,
                context=context,
                auto_train=True
            )
        
        return assistant_message
    
    def stream_chat(self, user_message: str, use_rag: bool = True):
        """Stream response with automated RLHF"""
        
        greetings = ['hi', 'hey', 'hello', 'hola', 'yo', 'sup', 'wassup']
        is_greeting = user_message.lower().strip() in greetings
        
        sources = []
        context = ""
        
        if use_rag and not is_greeting:
            context, sources = self.get_relevant_context(user_message)
            
            if context:
                prompt = f"""Use this information to answer:

{context}

Question: {user_message}

Answer naturally without mentioning the context."""
            else:
                prompt = user_message
        else:
            prompt = user_message
        
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history[-10:]
        ]
        
        # Stream response from Groq
        full_response = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token
        
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            error_msg = "I apologize, but I encountered an error."
            full_response = error_msg
            yield error_msg
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
        
        # Automated RLHF
        if self.enable_rlhf:
            self.rlhf_system.process_interaction(
                question=user_message,
                response=full_response,
                context=context,
                auto_train=True
            )
    
    def show_rlhf_stats(self):
        """Display RLHF statistics"""
        if self.enable_rlhf:
            self.rlhf_system.get_statistics()
        else:
            print("RLHF is not enabled")
    
    def show_improvements(self):
        """Show improvement suggestions"""
        if self.enable_rlhf:
            self.rlhf_system.get_improvement_suggestions()
        else:
            print("RLHF is not enabled")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("✓ Conversation history cleared")
    
    def save_conversation(self, filename: str = None):
        """Save conversation to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for msg in self.conversation_history:
                role = msg['role'].upper()
                content = msg['content']
                f.write(f"{role}:\n{content}\n\n")
        
        logger.info(f"✓ Conversation saved to {filename}")


def interactive_chat():
    """Run interactive chatbot with automated RLHF"""
    
    print("\n" + "="*80)
    print("FLOWBOTICS AI CHATBOT (Groq API + Automated RLHF)")
    print("="*80)
    print("\nCommands:")
    print("  'stats'    - Show RLHF training statistics")
    print("  'improve'  - Show improvement suggestions")
    print("  'clear'    - Clear conversation history")
    print("  'save'     - Save conversation")
    print("  'quit'     - Exit")
    print("="*80 + "\n")
    
    # Initialize chatbot
    try:
        chatbot = FlowboticsChatbotOptimized(
            model_name="llama-3.3-70b-versatile",
            enable_rlhf=True
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Get API key from: https://console.groq.com/keys")
        print("2. Set GROQ_API_KEY constant at the top of this file")
        return
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                chatbot.show_rlhf_stats()
                continue
            
            if user_input.lower() == 'improve':
                chatbot.show_improvements()
                continue
            
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                print("✓ Conversation cleared!\n")
                continue
            
            if user_input.lower() == 'save':
                chatbot.save_conversation()
                continue
            
            # Get streaming response
            print("\nAssistant: ", end="", flush=True)
            for token in chatbot.stream_chat(user_input):
                print(token, end="", flush=True)
            print("\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")


# if __name__ == "__main__":
#     # Check if API key is set
#     if not api_key:
#         print("\n" + "="*80)
#         print("⚠️  GROQ_API_KEY not set")
#         print("="*80)
#         print("\nSetup Instructions:")
#         print("1. Get your API key: https://console.groq.com/keys")
#         print("2. Open this file and set GROQ_API_KEY at the top:")
#         print('   GROQ_API_KEY = "your-key-here"')
#         print("="*80 + "\n")
#     else:
#         # Start interactive chatbot
#         interactive_chat()

if __name__ == "__main__":
    # Start interactive chatbot
    try:
        interactive_chat()
    except Exception as e:
        print(f"Error: {e}")
