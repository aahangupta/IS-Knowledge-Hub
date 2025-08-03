"""
Test script for Supabase integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.database import SupabaseManager, ISCode, DocumentChunkDB
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def cleanup_test_data(manager: SupabaseManager, code_number: str, version: str):
    """Clean up test data by code number and version"""
    print(f"\nCleaning up test data for code: {code_number} ({version})")
    try:
        # First, get the ID of the test code
        test_code = manager.get_is_code_by_number(code_number, version)
        if test_code and test_code.id:
            # Deleting the IS code will cascade delete the chunks
            response = manager.client.table('is_codes').delete().eq('id', test_code.id).execute()
            print(f"Cleanup response: {response}")
        else:
            print("Test code not found, no cleanup needed.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def test_supabase_integration():
    """Test full Supabase integration workflow"""
    print("\n" + "=" * 60)
    print("Testing Supabase Integration")
    print("=" * 60)
    
    print("\nNote: This test requires SUPABASE_URL and SUPABASE_KEY in .env file")
    
    manager = SupabaseManager()
    test_code_number = "IS TEST 1"
    test_code_version = "2025"
    
    try:
        # 1. Cleanup previous test data
        cleanup_test_data(manager, test_code_number, test_code_version)
        
        # 2. Add a new IS code
        print("\nAdding a new IS code...")
        test_code = ISCode(
            code_number=test_code_number,
            title="Test IS Code for Integration",
            version=test_code_version
        )
        added_code = manager.add_is_code(test_code)
        assert added_code is not None
        assert added_code.id is not None
        print(f"Added IS code with ID: {added_code.id}")
        
        # Keep track of the ID
        test_code_id = added_code.id
        
        # 3. Get the IS code by ID
        print("\nGetting IS code by ID...")
        retrieved_code = manager.get_is_code_by_id(test_code_id)
        assert retrieved_code is not None
        assert retrieved_code.title == test_code.title
        print("Successfully retrieved IS code by ID")
        
        # 4. Update the IS code status
        print("\nUpdating IS code status...")
        updated_code = manager.update_is_code_status(test_code_id, "processed")
        assert updated_code is not None
        assert updated_code.status == "processed"
        print("Successfully updated IS code status")
        
        # 5. Add document chunks
        print("\nAdding document chunks...")
        test_chunks = [
            DocumentChunkDB(
                is_code_id=test_code_id,
                chunk_id="chunk_1",
                content="This is the first test chunk."
            ),
            DocumentChunkDB(
                is_code_id=test_code_id,
                chunk_id="chunk_2",
                content="This is the second test chunk."
            )
        ]
        added_count = manager.add_document_chunks(test_chunks)
        assert added_count == 2
        print(f"Added {added_count} chunks")
        
        # 6. Get chunks for the IS code
        print("\nGetting chunks for IS code...")
        retrieved_chunks = manager.get_chunks_for_is_code(test_code_id)
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].content == test_chunks[0].content
        print("Successfully retrieved chunks")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure SUPABASE_URL and SUPABASE_KEY are set in .env")
        
    finally:
        # Final cleanup
        cleanup_test_data(manager, test_code_number, test_code_version)
        
        print("\n✅ Supabase integration test completed!")

if __name__ == "__main__":
    test_supabase_integration()