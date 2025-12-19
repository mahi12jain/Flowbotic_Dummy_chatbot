"""
chatbot_llm.py
Groq API Chatbot with Safe Imports for Streamlit Deployment
"""

from groq import Groq
from datetime import datetime
import logging
import os

# Safe imports with fallbacks
try:
    from Vector_dataset import VectorDBStore
    VECTORDB_AVAILABLE = True
except Exception as e:
    logging.warning(f"VectorDB not available: {e}")
    VECTORDB_AVAILABLE = False
    
try:
    from RLFH_feedback import AutomatedRLHFSystem
    RLHF_AVAILABLE = True
except Exception as e:
    logging.warning(f"RLHF not available: {e}")
    RLHF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowboticsChatbotOptimized:
    """
    Streamlit-safe chatbot with graceful degradation
    """
    
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile", 
                 persist_directory: str = "./chroma_db", enable_rlhf=True):
        """
        Initialize optimized chatbot with Groq API
        
        Args:
            api_key: Groq API key
            model_name: Groq model name
            persist_directory: ChromaDB storage path
            enable_rlhf: Enable automated RLHF training
        """
        
        # Validate API key
        self.api_key = api_key 
        if not self.api_key:
            raise ValueError("Groq API key required!")
        
        # Initialize Groq client
        try:
            self.client = Groq(api_key=self.api_key)
            self.model_name = model_name
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client: {e}")
        
        # Initialize VectorDB (optional)
        self.vectordb = None
        if VECTORDB_AVAILABLE:
            try:
                self.vectordb = VectorDBStore(persist_directory=persist_directory)
                logger.info(f"✓ VectorDB loaded: {self.vectordb.get_stats()} chunks")
            except Exception as e:
                logger.warning(f"VectorDB initialization failed: {e}")
                self.vectordb = None
        
        # Initialize RLHF (optional)
        self.rlhf_system = None
        self.enable_rlhf = enable_rlhf and RLHF_AVAILABLE
        if self.enable_rlhf:
            try:
                self.rlhf_system = AutomatedRLHFSystem()
                logger.info("✓ RLHF enabled")
            except Exception as e:
                logger.warning(f"RLHF initialization failed: {e}")
                self.rlhf_system = None
                self.enable_rlhf = False
        
        # Conversation memory
        self.conversation_history = []
        
        # System prompt
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

Response Strategy:
1. Listen: Acknowledge what the user shares (1 sentence)
2. Ask: ONE specific binary or limited-choice question
3. Wait: Let them answer before explaining solutions
4. Recommend: After understanding, suggest ONE specific solution
5. Guide: Provide clear next step

Response Guidelines:
- Keep responses SHORT (2-3 sentences maximum)
- First sentence: Acknowledge/affirm
- Second sentence: ONE specific question with 2-3 clear options
- Don't explain features until you understand their specific problem
- Use "or" to create binary choices: "X or Y?"
- Make questions easy to answer (not open-ended)
- Questions must be directly relevant to their previous answer
- Never ask multiple things in one question
- Always connect features to the user's specific needs
- Use concrete examples relevant to their industry
- End with a clear question or call-to-action
- NEVER list all services/features unless specifically asked
- Focus on relevance, not completeness"""

        logger.info(f"✓ Chatbot initialized with Groq API")
        logger.info(f"✓ Model: {model_name}")
        logger.info(f"✓ VectorDB: {'Available' if self.vectordb else 'Disabled'}")
        logger.info(f"✓ RLHF: {'Enabled' if self.enable_rlhf else 'Disabled'}")
    
    def get_relevant_context(self, question: str, n_results: int = 3) -> tuple:
        """Retrieve relevant context from VectorDB (if available)"""
        if not self.vectordb:
            return "", []
        
        try:
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
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return "", []
    
    def chat(self, user_message: str, use_rag: bool = True) -> str:
        """
        Generate response with optional RAG
        
        Args:
            user_message: User's question
            use_rag: Whether to use RAG (if available)
        
        Returns:
            Assistant's response
        """
        
        # Check for casual greeting
        greetings = ['hi', 'hey', 'hello', 'hola', 'yo', 'sup', 'wassup']
        is_greeting = user_message.lower().strip() in greetings
        
        sources = []
        context = ""
        
        # Build prompt with RAG (if available)
        if use_rag and not is_greeting and self.vectordb:
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
            *self.conversation_history[-10:]  # Keep last 10 exchanges
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
        
        # Process with RLHF (if available)
        if self.enable_rlhf and self.rlhf_system:
            try:
                self.rlhf_system.process_interaction(
                    question=user_message,
                    response=assistant_message,
                    context=context,
                    auto_train=True
                )
            except Exception as e:
                logger.warning(f"RLHF processing failed: {e}")
        
        return assistant_message
    
    def stream_chat(self, user_message: str, use_rag: bool = True):
        """Stream response with optional RAG"""
        
        greetings = ['hi', 'hey', 'hello', 'hola', 'yo', 'sup', 'wassup']
        is_greeting = user_message.lower().strip() in greetings
        
        sources = []
        context = ""
        
        # Build prompt with RAG (if available)
        if use_rag and not is_greeting and self.vectordb:
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
        
        # Process with RLHF (if available)
        if self.enable_rlhf and self.rlhf_system:
            try:
                self.rlhf_system.process_interaction(
                    question=user_message,
                    response=full_response,
                    context=context,
                    auto_train=True
                )
            except Exception as e:
                logger.warning(f"RLHF processing failed: {e}")
    
    def show_rlhf_stats(self):
        """Display RLHF statistics"""
        if self.enable_rlhf and self.rlhf_system:
            self.rlhf_system.get_statistics()
        else:
            print("RLHF is not available")
    
    def show_improvements(self):
        """Show improvement suggestions"""
        if self.enable_rlhf and self.rlhf_system:
            self.rlhf_system.get_improvement_suggestions()
        else:
            print("RLHF is not available")
    
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
