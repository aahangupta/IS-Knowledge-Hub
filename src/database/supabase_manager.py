"""
Supabase Database Integration
Handles connection and data operations for the Supabase project
"""

import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from config import settings
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ISCode:
    """
    Data model for an IS code document
    """
    id: Optional[str] = None
    code_number: str = ""
    title: str = ""
    version: Optional[str] = None
    pdf_path: Optional[str] = None
    markdown_path: Optional[str] = None
    uploaded_at: Optional[str] = None
    processed_at: Optional[str] = None
    status: str = "pending"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DocumentChunkDB:
    """
    Data model for a document chunk in the database
    """
    id: Optional[str] = None
    is_code_id: str = ""
    chunk_id: str = ""
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    embedded_at: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class SupabaseManager:
    """
    Manages Supabase connection and data operations
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase manager
        """
        self.url = url or settings.supabase_url
        self.key = key or settings.supabase_key
        
        # Initialize Supabase client
        self.client: Client = create_client(self.url, self.key)
        logger.info("Initialized SupabaseManager")
        
    # --- IS Code Operations ---
    
    def add_is_code(self, code: ISCode) -> Optional[ISCode]:
        """
        Add a new IS code to the database
        """
        data = code.to_dict()
        del data['id'] # Let Supabase generate the ID
        
        response = self.client.table('is_codes').insert(data).execute()
        
        if response.data:
            return ISCode(**response.data[0])
        
        logger.error(f"Error adding IS code: {response.error}")
        return None
        
    def get_is_code_by_id(self, code_id: str) -> Optional[ISCode]:
        """
        Get an IS code by its ID
        """
        response = self.client.table('is_codes').select("*").eq('id', code_id).execute()
        
        if response.data:
            return ISCode(**response.data[0])
            
        return None
        
    def get_is_code_by_number(self, code_number: str, version: Optional[str] = None) -> Optional[ISCode]:
        """
        Get an IS code by its number and optional version
        """
        query = self.client.table('is_codes').select("*").eq('code_number', code_number)
        
        if version:
            query = query.eq('version', version)
            
        response = query.execute()
        
        if response.data:
            return ISCode(**response.data[0])
            
        return None
        
    def list_is_codes(self) -> List[ISCode]:
        """
        List all IS codes
        """
        response = self.client.table('is_codes').select("*").execute()
        return [ISCode(**item) for item in response.data]
        
    def update_is_code_status(self, code_id: str, status: str) -> Optional[ISCode]:
        """
        Update the status of an IS code
        """
        response = self.client.table('is_codes').update({'status': status}).eq('id', code_id).execute()
        
        if response.data:
            return ISCode(**response.data[0])
            
        return None
        
    # --- Document Chunk Operations ---
    
    def add_document_chunks(self, chunks: List[DocumentChunkDB]) -> int:
        """
        Add multiple document chunks to the database
        """
        data = [chunk.to_dict() for chunk in chunks]
        for item in data:
            del item['id']
            
        response = self.client.table('document_chunks').insert(data).execute()
        
        if response.data:
            return len(response.data)
            
        return 0
        
    def get_chunks_for_is_code(self, code_id: str) -> List[DocumentChunkDB]:
        """
        Get all chunks for a specific IS code
        """
        response = self.client.table('document_chunks').select("*").eq('is_code_id', code_id).execute()
        return [DocumentChunkDB(**item) for item in response.data]
