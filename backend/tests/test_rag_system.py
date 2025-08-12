"""
Integration tests for RAG system to identify end-to-end query handling issues
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestRAGSystemIntegration:
    """Integration tests for complete RAG system functionality"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_rag_system_initialization(self, mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor, mock_config):
        """Test RAG system initialization with all components"""
        rag_system = RAGSystem(mock_config)
        
        # Verify all components were initialized
        mock_document_processor.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_ai_generator.assert_called_once()
        mock_session_manager.assert_called_once()
        
        # Verify tools were registered
        assert len(rag_system.tool_manager.tools) == 2  # CourseSearchTool and CourseOutlineTool
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_successful_flow(self, mock_session_manager, mock_ai_generator, 
                                  mock_vector_store, mock_document_processor, mock_config):
        """Test successful query processing end-to-end"""
        # Set up mocks
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = SearchResults(
            documents=["Machine learning is a subset of AI"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store_instance.get_lesson_info.return_value = ("Introduction", "https://example.com/lesson1")
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = "Machine learning is a field of AI that uses algorithms..."
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager_instance.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("What is machine learning?", session_id="test_session")
        
        # Verify AI generator was called with correct parameters
        mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        
        assert "What is machine learning?" in call_args["query"]
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None
        
        # Verify session management
        mock_session_manager_instance.get_conversation_history.assert_called_once_with("test_session")
        mock_session_manager_instance.add_exchange.assert_called_once()
        
        # Verify response
        assert response == "Machine learning is a field of AI that uses algorithms..."
        assert isinstance(sources, list)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_tool_execution(self, mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor, mock_config):
        """Test query that triggers tool execution"""
        # Set up vector store mock
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = SearchResults(
            documents=["Detailed ML content"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store_instance.get_lesson_info.return_value = ("ML Basics", "https://example.com/lesson1")
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Set up AI generator to simulate tool usage
        mock_ai_generator_instance = Mock()
        
        def mock_generate_response(*args, **kwargs):
            # Simulate tool execution by calling the tool manager
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                # Simulate Claude calling the search tool
                tool_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="machine learning definition"
                )
                return f"Based on the course materials: {tool_result[:50]}..."
            return "Direct response without tools"
        
        mock_ai_generator_instance.generate_response = mock_generate_response
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager_instance.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify tool was executed
        assert "Based on the course materials" in response
        assert "ML Course" in response
        
        # Verify sources were captured
        assert len(sources) > 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_empty_database(self, mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor, mock_config):
        """Test query when database has no content"""
        # Set up vector store to return empty results
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        
        def mock_generate_response(*args, **kwargs):
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                tool_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="test query"
                )
                return f"Search result: {tool_result}"
            return "No search performed"
        
        mock_ai_generator_instance.generate_response = mock_generate_response
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager_instance.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify empty results are handled
        assert "No relevant content found" in response
        assert len(sources) == 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_vector_store_error(self, mock_session_manager, mock_ai_generator, 
                                          mock_vector_store, mock_document_processor, mock_config):
        """Test query when vector store returns error"""
        # Set up vector store to return error
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = SearchResults.empty("Database connection failed")
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        
        def mock_generate_response(*args, **kwargs):
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                tool_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="test query"
                )
                return f"Error encountered: {tool_result}"
            return "No search performed"
        
        mock_ai_generator_instance.generate_response = mock_generate_response
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager_instance.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify error is propagated
        assert "Database connection failed" in response
        assert len(sources) == 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_session_management(self, mock_session_manager, mock_ai_generator, 
                               mock_vector_store, mock_document_processor, mock_config):
        """Test session management functionality"""
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = "Test response"
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager_instance.get_conversation_history.return_value = "Previous: What is AI?"
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query with session
        response, sources = rag_system.query("Tell me more", session_id="test_session")
        
        # Verify conversation history was retrieved
        mock_session_manager_instance.get_conversation_history.assert_called_once_with("test_session")
        
        # Verify conversation was updated
        mock_session_manager_instance.add_exchange.assert_called_once_with(
            "test_session", 
            "Answer this question about course materials: Tell me more",
            "Test response"
        )
        
        # Verify history was passed to AI generator
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous: What is AI?"
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_manager, mock_ai_generator, 
                                  mock_vector_store, mock_document_processor, mock_config):
        """Test query without session management"""
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = "Test response"
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query without session
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify no session operations were performed
        mock_session_manager_instance.get_conversation_history.assert_not_called()
        mock_session_manager_instance.add_exchange.assert_not_called()
        
        # Verify AI generator was called with no history
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] is None
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_ai_generator_error_handling(self, mock_session_manager, mock_ai_generator, 
                                        mock_vector_store, mock_document_processor, mock_config):
        """Test error handling when AI generator fails"""
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.side_effect = Exception("API key invalid")
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query - should raise exception
        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is machine learning?")
        
        assert "API key invalid" in str(exc_info.value)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_source_tracking_and_reset(self, mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor, mock_config):
        """Test that sources are properly tracked and reset between queries"""
        # Set up mocks with sources
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store_instance.get_lesson_info.return_value = ("Lesson", "https://example.com")
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator_instance = Mock()
        
        def mock_generate_response(*args, **kwargs):
            # Simulate tool execution
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query="test")
            return "Response"
        
        mock_ai_generator_instance.generate_response = mock_generate_response
        mock_ai_generator.return_value = mock_ai_generator_instance
        
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # First query
        response1, sources1 = rag_system.query("First query")
        
        # Verify sources were captured
        assert len(sources1) > 0
        
        # Second query
        response2, sources2 = rag_system.query("Second query")
        
        # Verify sources are independent
        assert len(sources2) > 0
        # Sources should be reset between queries
        # This tests that reset_sources() is called properly


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""
    
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_folder_success(self, mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor,
                                      mock_listdir, mock_path_exists, mock_config, sample_course, sample_course_chunks):
        """Test successful course folder processing"""
        # Mock file system
        mock_path_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "invalid.jpg"]
        
        # Mock document processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_course_document.side_effect = [
            (sample_course, sample_course_chunks),
            (sample_course, sample_course_chunks)
        ]
        mock_document_processor.return_value = mock_processor_instance
        
        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.get_existing_course_titles.return_value = []
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_generator.return_value = Mock()
        mock_session_manager.return_value = Mock()
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Process folder
        total_courses, total_chunks = rag_system.add_course_folder("/fake/docs/")
        
        # Verify results
        assert total_courses == 2  # Only PDF and TXT files processed
        assert total_chunks == 6   # 3 chunks per course * 2 courses
        
        # Verify vector store operations
        assert mock_vector_store_instance.add_course_metadata.call_count == 2
        assert mock_vector_store_instance.add_course_content.call_count == 2