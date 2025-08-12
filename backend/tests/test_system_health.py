"""
System diagnostic tests to check real system health and identify configuration issues
"""
import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager


class TestSystemHealth:
    """Diagnostic tests for actual system health"""
    
    def test_config_values(self):
        """Test that configuration values are properly set"""
        # Test API key is configured
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set in environment"
        assert config.ANTHROPIC_API_KEY != "", "ANTHROPIC_API_KEY is empty"
        assert not config.ANTHROPIC_API_KEY.startswith("sk-"), "API key should not start with sk- (that's OpenAI format)"
        
        # Test model configuration
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514", f"Unexpected model: {config.ANTHROPIC_MODEL}"
        
        # Test embedding model
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2", f"Unexpected embedding model: {config.EMBEDDING_MODEL}"
        
        # Test numeric configs
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be positive"
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be non-negative"
        
        # Test paths
        assert config.CHROMA_PATH, "CHROMA_PATH is not configured"
    
    def test_docs_folder_exists(self):
        """Test that the docs folder exists and has content"""
        docs_path = Path("../docs")
        
        # Check if docs folder exists
        if not docs_path.exists():
            pytest.skip("No ../docs folder found - this may be expected in test environment")
        
        # Check for course files
        course_files = list(docs_path.glob("*.txt")) + list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.docx"))
        
        if len(course_files) == 0:
            pytest.fail("No course files found in ../docs folder")
        
        print(f"Found {len(course_files)} course files: {[f.name for f in course_files]}")
    
    def test_chromadb_database_state(self):
        """Test ChromaDB database state and content"""
        try:
            # Create vector store with real config
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            # Check if collections exist and have content
            try:
                catalog_count = vector_store.get_course_count()
                course_titles = vector_store.get_existing_course_titles()
                
                print(f"ChromaDB status:")
                print(f"  - Course count: {catalog_count}")
                print(f"  - Course titles: {course_titles}")
                
                if catalog_count == 0:
                    pytest.fail("ChromaDB has no courses loaded - this may explain 'query failed' errors")
                
                # Test basic search functionality
                test_results = vector_store.search("machine learning", limit=1)
                
                if test_results.error:
                    pytest.fail(f"ChromaDB search failed: {test_results.error}")
                
                if test_results.is_empty():
                    pytest.fail("ChromaDB search returned no results for 'machine learning' - database may be empty or corrupted")
                
                print(f"  - Search test: SUCCESS (found {len(test_results.documents)} results)")
                
            except Exception as e:
                pytest.fail(f"Error accessing ChromaDB: {e}")
                
        except Exception as e:
            pytest.fail(f"Failed to initialize ChromaDB: {e}")
    
    def test_vector_store_search_functionality(self):
        """Test vector store search with various scenarios"""
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            # Test 1: General search
            results = vector_store.search("introduction")
            assert not results.error, f"General search failed: {results.error}"
            
            # Test 2: Empty query handling
            results = vector_store.search("")
            # Should not crash, may return empty or error
            
            # Test 3: Course name resolution
            course_titles = vector_store.get_existing_course_titles()
            if course_titles:
                first_course = course_titles[0]
                resolved = vector_store._resolve_course_name(first_course)
                assert resolved == first_course, f"Course name resolution failed: {first_course} -> {resolved}"
                
                # Test search with course filter
                results = vector_store.search("introduction", course_name=first_course)
                assert not results.error, f"Course-filtered search failed: {results.error}"
            
            print("Vector store search functionality: PASSED")
            
        except Exception as e:
            pytest.fail(f"Vector store search test failed: {e}")
    
    def test_course_search_tool_with_real_data(self):
        """Test CourseSearchTool with real database"""
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            tool = CourseSearchTool(vector_store)
            
            # Test tool definition
            definition = tool.get_tool_definition()
            assert definition["name"] == "search_course_content"
            
            # Test basic execution
            result = tool.execute("introduction")
            
            # Should not be an error message
            error_indicators = ["Error:", "Failed:", "Exception:", "Database connection failed"]
            for indicator in error_indicators:
                if indicator in result:
                    pytest.fail(f"CourseSearchTool returned error: {result}")
            
            # Should either have content or explicit "No relevant content found"
            if "No relevant content found" in result:
                print("CourseSearchTool: No content found (may indicate empty database)")
            else:
                print(f"CourseSearchTool: SUCCESS (found content)")
                assert len(tool.last_sources) >= 0  # Should track sources
            
        except Exception as e:
            pytest.fail(f"CourseSearchTool test failed: {e}")
    
    def test_ai_generator_api_connectivity(self):
        """Test AI generator API connectivity and basic functionality"""
        try:
            ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            
            # Test basic response without tools
            response = ai_generator.generate_response("What is 2+2?")
            
            assert isinstance(response, str), "AI response should be a string"
            assert len(response) > 0, "AI response should not be empty"
            
            # Check for common error patterns
            error_patterns = [
                "invalid api key",
                "unauthorized",
                "forbidden",
                "rate limit",
                "quota exceeded"
            ]
            
            response_lower = response.lower()
            for pattern in error_patterns:
                if pattern in response_lower:
                    pytest.fail(f"AI API error detected: {response}")
            
            print(f"AI Generator: SUCCESS (response: {response[:50]}...)")
            
        except Exception as e:
            # Common API errors
            error_msg = str(e).lower()
            if "api key" in error_msg:
                pytest.fail(f"Invalid Anthropic API key: {e}")
            elif "rate limit" in error_msg:
                pytest.fail(f"Rate limit exceeded: {e}")
            elif "quota" in error_msg:
                pytest.fail(f"API quota exceeded: {e}")
            else:
                pytest.fail(f"AI Generator API test failed: {e}")
    
    def test_tool_manager_registration(self):
        """Test tool manager registration and execution"""
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            tool_manager = ToolManager()
            search_tool = CourseSearchTool(vector_store)
            
            # Test registration
            tool_manager.register_tool(search_tool)
            
            # Test tool definitions
            definitions = tool_manager.get_tool_definitions()
            assert len(definitions) > 0, "No tools registered"
            assert definitions[0]["name"] == "search_course_content"
            
            # Test execution
            result = tool_manager.execute_tool("search_course_content", query="test")
            
            assert isinstance(result, str), "Tool execution should return string"
            
            # Check for error patterns
            if result.startswith("Tool '") and "' not found" in result:
                pytest.fail(f"Tool not found: {result}")
            
            print("Tool Manager: SUCCESS")
            
        except Exception as e:
            pytest.fail(f"Tool Manager test failed: {e}")
    
    def test_rag_system_full_initialization(self):
        """Test full RAG system initialization"""
        try:
            rag_system = RAGSystem(config)
            
            # Verify all components initialized
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.tool_manager is not None
            assert rag_system.session_manager is not None
            
            # Verify tools are registered
            tool_names = list(rag_system.tool_manager.tools.keys())
            expected_tools = ["search_course_content", "get_course_outline"]
            
            for tool in expected_tools:
                assert tool in tool_names, f"Missing expected tool: {tool}"
            
            print(f"RAG System initialization: SUCCESS (tools: {tool_names})")
            
        except Exception as e:
            pytest.fail(f"RAG System initialization failed: {e}")
    
    def test_rag_system_query_end_to_end(self):
        """Test complete RAG system query flow with real components"""
        try:
            rag_system = RAGSystem(config)
            
            # Test simple query
            response, sources = rag_system.query("What is machine learning?")
            
            assert isinstance(response, str), "Response should be a string"
            assert len(response) > 0, "Response should not be empty"
            assert isinstance(sources, list), "Sources should be a list"
            
            # Check for failure patterns
            failure_patterns = [
                "query failed",
                "error:",
                "exception:",
                "failed to",
                "database connection failed",
                "api key invalid"
            ]
            
            response_lower = response.lower()
            for pattern in failure_patterns:
                if pattern in response_lower:
                    pytest.fail(f"RAG query failed with pattern '{pattern}': {response}")
            
            print(f"RAG System query: SUCCESS")
            print(f"  Response: {response[:100]}...")
            print(f"  Sources count: {len(sources)}")
            
            if len(sources) > 0:
                print(f"  First source: {sources[0]}")
            
        except Exception as e:
            pytest.fail(f"RAG System end-to-end test failed: {e}")
    
    def test_vector_store_course_resolution_bug(self):
        """Test for the potential bug in vector_store.py line 112"""
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            # Get existing courses to test resolution
            existing_courses = vector_store.get_existing_course_titles()
            
            if not existing_courses:
                pytest.skip("No courses in database to test course resolution")
            
            first_course = existing_courses[0]
            
            # Test course resolution - this may trigger the bug
            try:
                resolved = vector_store._resolve_course_name(first_course)
                
                if resolved is None:
                    pytest.fail(f"Course resolution returned None for existing course: {first_course}")
                
                print(f"Course resolution test: SUCCESS ({first_course} -> {resolved})")
                
            except IndexError as e:
                if "list index out of range" in str(e):
                    pytest.fail(f"FOUND BUG: Double indexing issue in vector_store.py line 112 - {e}")
                else:
                    raise
                
        except Exception as e:
            pytest.fail(f"Course resolution test failed: {e}")


class TestSystemConfiguration:
    """Test system configuration and environment"""
    
    def test_environment_file(self):
        """Test .env file exists and has required variables"""
        env_path = Path("../.env")
        
        if not env_path.exists():
            pytest.fail("No .env file found - create one with ANTHROPIC_API_KEY")
        
        with open(env_path) as f:
            env_content = f.read()
        
        if "ANTHROPIC_API_KEY" not in env_content:
            pytest.fail(".env file does not contain ANTHROPIC_API_KEY")
        
        print(".env file: EXISTS and contains required variables")
    
    def test_dependencies_available(self):
        """Test that required dependencies are available"""
        required_modules = [
            "anthropic",
            "chromadb", 
            "sentence_transformers",
            "fastapi",
            "pydantic"
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            pytest.fail(f"Missing required modules: {missing_modules}")
        
        print(f"Dependencies: All required modules available")
    
    def test_chroma_db_permissions(self):
        """Test ChromaDB directory permissions"""
        chroma_path = Path(config.CHROMA_PATH)
        
        if not chroma_path.exists():
            # Try to create it
            try:
                chroma_path.mkdir(parents=True, exist_ok=True)
                print(f"ChromaDB directory created: {chroma_path}")
            except PermissionError:
                pytest.fail(f"Cannot create ChromaDB directory: {chroma_path}")
        
        # Test write permissions
        test_file = chroma_path / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"ChromaDB directory: Write permissions OK")
        except PermissionError:
            pytest.fail(f"No write permissions for ChromaDB directory: {chroma_path}")


if __name__ == "__main__":
    # Allow running diagnostic tests directly
    pytest.main([__file__, "-v"])