"""
Prompt Manager for RAG System
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages the creation of prompts for the RAG system
    """
    
    def __init__(self, system_prompt: str = None):
        """
        Initialize prompt manager
        
        Args:
            system_prompt: Default system prompt
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt
        """
        return """
You are an expert assistant for Indian Standards (IS) codes in the construction industry.
Your role is to answer questions accurately based on the provided context from IS code documents.

Instructions:
1.  Analyze the user's question carefully.
2.  Use the provided context to formulate your answer.
3.  Cite the source of your information (e.g., "According to IS 456:2000, Clause 5.1...").
4.  If the context does not contain the answer, state that clearly. Do not make up information.
5.  Keep your answers concise and directly related to the question.
6.  If you need to perform calculations, explain your steps clearly.
7.  If the question is ambiguous, ask for clarification.
"""
        
    def create_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Create a prompt for the language model
        
        Args:
            query: User's question
            context: Retrieved context from IS codes
            
        Returns:
            A list of messages for the chat completions API
        """
        user_prompt = f"""
        User Question:
        "{query}"

        Context from IS Codes:
        ---
        {context}
        ---

        Based on the provided context, please answer the user's question.
        Remember to cite the source of your information.
        """
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.debug(f"Created prompt for query: {query}")
        return messages
        
    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt
        
        Args:
            new_prompt: The new system prompt
        """
        self.system_prompt = new_prompt
        logger.info("System prompt updated")