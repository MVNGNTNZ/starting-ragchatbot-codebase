"""
Test fixtures and configuration for RAG chatbot tests
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is Machine Learning?", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Linear Regression", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Classification", lesson_link="https://example.com/lesson3")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Linear regression is a statistical method for modeling the relationship between variables.",
            course_title="Introduction to Machine Learning", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Classification algorithms predict categorical outcomes based on input features.",
            course_title="Introduction to Machine Learning",
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_search_results_success():
    """Create mock successful search results"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
            "Linear regression is a statistical method for modeling the relationship between variables."
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def mock_search_results_empty():
    """Create mock empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_search_results_error():
    """Create mock error search results"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    
    # Default successful search response
    mock_store.search.return_value = SearchResults(
        documents=["Sample content about machine learning"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}],
        distances=[0.1]
    )
    
    # Mock lesson info retrieval
    mock_store.get_lesson_info.return_value = ("Introduction to ML", "https://example.com/lesson1")
    
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response without tools
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response that uses tools"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "machine learning"}
    
    mock_response.content = [mock_tool_block]
    
    return mock_response


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "course_name": {"type": "string", "description": "Course title"},
                    "lesson_number": {"type": "integer", "description": "Lesson number"}
                },
                "required": ["query"]
            }
        }
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    config.MAX_TOOL_ROUNDS = 2
    return config


@pytest.fixture
def mock_sequential_tool_responses():
    """Mock Anthropic responses for sequential tool calling tests"""
    
    # Round 1: Claude requests a tool call
    round1_response = Mock()
    round1_response.stop_reason = "tool_use"
    
    round1_tool_block = Mock()
    round1_tool_block.type = "tool_use"
    round1_tool_block.name = "search_course_content"
    round1_tool_block.id = "tool_round1"
    round1_tool_block.input = {"query": "machine learning basics"}
    
    round1_response.content = [round1_tool_block]
    
    # Round 2: Claude requests another tool call  
    round2_response = Mock()
    round2_response.stop_reason = "tool_use"
    
    round2_tool_block = Mock()
    round2_tool_block.type = "tool_use"
    round2_tool_block.name = "search_course_content"
    round2_tool_block.id = "tool_round2"
    round2_tool_block.input = {"query": "neural networks"}
    
    round2_response.content = [round2_tool_block]
    
    # Round 3: Final response without tools
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_response.content = [Mock(text="Based on my searches, machine learning and neural networks are...")]
    
    return [round1_response, round2_response, final_response]


@pytest.fixture
def mock_early_termination_responses():
    """Mock responses where Claude terminates early (no second round)"""
    
    # Round 1: Claude requests a tool call
    round1_response = Mock()
    round1_response.stop_reason = "tool_use"
    
    round1_tool_block = Mock()
    round1_tool_block.type = "tool_use"
    round1_tool_block.name = "search_course_content"
    round1_tool_block.id = "tool_123"
    round1_tool_block.input = {"query": "python basics"}
    
    round1_response.content = [round1_tool_block]
    
    # Round 2: Claude provides final answer without tools
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_response.content = [Mock(text="Python is a programming language...")]
    
    return [round1_response, final_response]


@pytest.fixture
def mock_anthropic_multi_round_client():
    """Mock Anthropic client that supports sequential tool calling scenarios"""
    mock_client = Mock()
    
    def create_message_side_effect(*args, **kwargs):
        # This will be configured per test
        pass
    
    mock_client.messages.create.side_effect = create_message_side_effect
    return mock_client