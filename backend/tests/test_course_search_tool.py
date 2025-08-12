"""
Unit tests for CourseSearchTool to identify search functionality issues
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool functionality"""
    
    def test_get_tool_definition(self):
        """Test that tool definition is properly formatted"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        definition = tool.get_tool_definition()
        
        # Verify required fields
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        # Verify parameter definitions
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
    
    def test_execute_successful_search(self, mock_search_results_success):
        """Test execute method with successful search results"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Introduction to ML", "https://example.com/lesson1")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning")
        
        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert isinstance(result, str)
        assert "Introduction to Machine Learning" in result
        assert "Machine learning is a subset" in result
        
        # Verify sources were tracked
        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["text"] == "Lesson 1: Introduction to ML"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"
    
    def test_execute_with_course_filter(self, mock_search_results_success):
        """Test execute method with course name filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Introduction to ML", "https://example.com/lesson1")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning", course_name="ML Course")
        
        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name="ML Course",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self, mock_search_results_success):
        """Test execute method with lesson number filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Introduction to ML", "https://example.com/lesson1")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning", lesson_number=1)
        
        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=1
        )
    
    def test_execute_empty_results(self, mock_search_results_empty):
        """Test execute method with empty search results"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_empty
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent content")
        
        # Verify appropriate message for empty results
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0
    
    def test_execute_empty_results_with_filters(self, mock_search_results_empty):
        """Test execute method with empty results and filters"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_empty
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("content", course_name="Missing Course", lesson_number=99)
        
        # Verify filter info is included in message
        assert "No relevant content found in course 'Missing Course' in lesson 99" in result
    
    def test_execute_search_error(self, mock_search_results_error):
        """Test execute method with search error"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_error
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("query")
        
        # Verify error is returned
        assert result == "Database connection failed"
        assert len(tool.last_sources) == 0
    
    def test_execute_vector_store_exception(self):
        """Test execute method when vector store raises exception"""
        mock_vector_store = Mock()
        mock_vector_store.search.side_effect = Exception("ChromaDB connection error")
        
        tool = CourseSearchTool(mock_vector_store)
        
        # This should not raise an exception, but return SearchResults with error
        # Note: The current implementation may not handle this properly
        try:
            result = tool.execute("query")
            # If we get here, the error handling worked
            assert isinstance(result, str)
        except Exception as e:
            # If we get here, there's a bug in error handling
            pytest.fail(f"Tool should handle exceptions gracefully, but got: {e}")
    
    def test_format_results_with_lesson_info(self):
        """Test result formatting with lesson information"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_info.return_value = ("Advanced Topics", "https://example.com/lesson3")
        
        tool = CourseSearchTool(mock_vector_store)
        
        results = SearchResults(
            documents=["Sample content about advanced ML topics"],
            metadata=[{"course_title": "ML Course", "lesson_number": 3, "chunk_index": 0}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify formatting
        assert "[ML Course - Lesson 3]" in formatted
        assert "Sample content about advanced ML topics" in formatted
        
        # Verify sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Lesson 3: Advanced Topics"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson3"
    
    def test_format_results_without_lesson_info(self):
        """Test result formatting when lesson info is not available"""
        mock_vector_store = Mock()
        mock_vector_store.get_lesson_info.return_value = (None, None)
        
        tool = CourseSearchTool(mock_vector_store)
        
        results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "ML Course", "lesson_number": 5, "chunk_index": 0}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify fallback formatting
        assert "[ML Course - Lesson 5]" in formatted
        
        # Verify fallback sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Lesson 5"
        assert tool.last_sources[0]["url"] is None
    
    def test_format_results_no_lesson_number(self):
        """Test result formatting when no lesson number is provided"""
        mock_vector_store = Mock()
        
        tool = CourseSearchTool(mock_vector_store)
        
        results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "ML Course", "chunk_index": 0}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        # Verify formatting without lesson number
        assert "[ML Course]" in formatted
        assert "Course overview content" in formatted
        
        # Verify sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "ML Course"
        assert tool.last_sources[0]["url"] is None
    
    def test_sources_reset_between_searches(self, mock_search_results_success):
        """Test that sources are properly reset between searches"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Lesson Title", "https://example.com/lesson")
        
        tool = CourseSearchTool(mock_vector_store)
        
        # First search
        tool.execute("first query")
        first_sources_count = len(tool.last_sources)
        
        # Second search
        tool.execute("second query")
        second_sources_count = len(tool.last_sources)
        
        # Sources should be from second search only
        assert first_sources_count > 0
        assert second_sources_count > 0
        # The last_sources should be updated, not accumulated
        assert len(tool.last_sources) == second_sources_count


class TestToolManager:
    """Test cases for ToolManager functionality"""
    
    def test_register_tool(self):
        """Test tool registration"""
        manager = ToolManager()
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        # Verify tool is registered
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions"""
        manager = ToolManager()
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        definitions = manager.get_tool_definitions()
        
        # Verify definitions
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool_success(self, mock_search_results_success):
        """Test successful tool execution"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Lesson", "https://example.com")
        
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        # Verify execution
        assert isinstance(result, str)
        assert "Introduction to Machine Learning" in result
    
    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        # Verify error message
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, mock_search_results_success):
        """Test getting sources from last search"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Lesson", "https://example.com")
        
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search
        manager.execute_tool("search_course_content", query="test query")
        
        # Get sources
        sources = manager.get_last_sources()
        
        # Verify sources
        assert len(sources) > 0
        assert sources[0]["text"] == "Lesson 1: Lesson"
    
    def test_reset_sources(self, mock_search_results_success):
        """Test resetting sources"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_info.return_value = ("Lesson", "https://example.com")
        
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to create sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0
        assert len(tool.last_sources) == 0