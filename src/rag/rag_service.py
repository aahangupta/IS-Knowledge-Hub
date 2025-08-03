"""
RAG Service for Question Answering
"""

import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from config import settings
from ..search import SearchEngine
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class RAGService:
    """
    Handles the RAG pipeline: search -> prompt -> generate
    """
    
    def __init__(
        self,
        search_engine: SearchEngine,
        prompt_manager: PromptManager,
        llm_model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG service
        
        Args:
            search_engine: Instance of SearchEngine
            prompt_manager: Instance of PromptManager
            llm_model: The language model to use for generation
            api_key: OpenAI API key
        """
        self.search_engine = search_engine
        self.prompt_manager = prompt_manager
        self.llm_model = llm_model
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        
        logger.info(f"Initialized RAGService with model: {llm_model}")
        
    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        max_context_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            max_context_tokens: Max tokens for the context
            
        Returns:
            A dictionary with the answer and retrieved context
        """
        
        # 1. Retrieve context
        logger.info(f"Retrieving context for query: '{query}'")
        context = self.search_engine.get_context_for_query(
            query=query,
            top_k=top_k,
            max_tokens=max_context_tokens
        )
        
        if not context:
            return {
                "answer": "I could not find any relevant information in the IS codes to answer your question.",
                "context": "",
                "query": query
            }
            
        # 2. Create prompt
        messages = self.prompt_manager.create_prompt(query, context)
        
        # 3. Generate answer
        logger.info("Generating answer from LLM...")
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2, # Lower temperature for more factual answers
                max_tokens=1024,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated successfully")
            
            return {
                "answer": answer,
                "context": context,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error generating answer from LLM: {e}")
            return {
                "answer": "I encountered an error while generating the answer. Please try again.",
                "context": context,
                "query": query
            }