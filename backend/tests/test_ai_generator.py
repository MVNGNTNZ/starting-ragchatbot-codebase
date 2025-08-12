"""
Unit tests for AIGenerator to identify tool calling and AI integration issues
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from types import SimpleNamespace

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key-123"
        self.model = "claude-sonnet-4-20250514"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_initialization(self, mock_anthropic):
        """Test AIGenerator initialization"""
        generator = AIGenerator(self.api_key, self.model)
        
        # Verify Anthropic client was created with correct API key
        mock_anthropic.assert_called_once_with(api_key=self.api_key)
        
        # Verify configuration
        assert generator.model == self.model
        assert generator.base_params["model"] == self.model
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tool usage"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response("What is machine learning?")
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == self.model
        assert call_args["messages"] == [{"role": "user", "content": "What is machine learning?"}]
        assert generator.SYSTEM_PROMPT in call_args["system"]
        
        # Verify result
        assert result == "This is a test response"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a follow-up response")]
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(self.api_key, self.model)
        history = "Previous conversation context"
        result = generator.generate_response("Follow-up question", conversation_history=history)
        
        # Verify history is included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert history in call_args["system"]
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_usage(self, mock_anthropic, sample_tool_definitions):
        """Test response generation with tools provided but not used"""
        # Mock response without tool usage
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without searching")]
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "What is 2+2?",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were provided in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == sample_tool_definitions
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Verify tool manager wasn't called since no tools were used
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Verify result
        assert result == "Direct answer without searching"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_usage(self, mock_anthropic, sample_tool_definitions):
        """Test response generation with actual tool usage"""
        # Mock initial response with tool usage
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock tool use content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning basics"}
        
        mock_tool_response.content = [mock_tool_block]
        
        # Mock final response after tool execution (now without tool_use stop_reason)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on the search results, machine learning is...")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: ML is a subset of AI..."
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "What is machine learning?",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )
        
        # Verify final API call was made
        assert mock_client.messages.create.call_count == 2
        
        # Verify result
        assert result == "Based on the search results, machine learning is..."
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic, sample_tool_definitions):
        """Test error handling when tool execution fails"""
        # Mock initial response with tool usage
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        
        mock_tool_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="I encountered an error while searching.")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager with error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Error: Database connection failed"
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Search for something",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was called and error was handled
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify second API call includes tool error result
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have user message, assistant tool use, and user tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert "Error: Database connection failed" in str(messages[2]["content"])
        
        # Verify final result
        assert result == "I encountered an error while searching."
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic, sample_tool_definitions):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool calls
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        # Create multiple tool blocks
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "first query"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"query": "second query"}
        
        mock_tool_response.content = [mock_tool_block1, mock_tool_block2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Combined results from both searches")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search results",
            "Second search results"
        ]
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Search for multiple things",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="second query")
        
        # Verify final result
        assert result == "Combined results from both searches"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(self.api_key, self.model)
        
        # Now errors are handled gracefully and returned as error messages
        result = generator.generate_response("Test query")
        
        assert "Error:" in result
        assert "API rate limit exceeded" in result
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_invalid_api_key(self, mock_anthropic):
        """Test handling of invalid API key"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Invalid API key")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(self.api_key, self.model)
        
        # Now errors are handled gracefully and returned as error messages
        result = generator.generate_response("Test query")
        
        assert "Error:" in result
        assert "Invalid API key" in result
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance"""
        generator = AIGenerator(self.api_key, self.model)
        
        prompt = generator.SYSTEM_PROMPT
        
        # Verify key instructions are present
        assert "course materials" in prompt.lower()
        assert "search" in prompt.lower()
        assert "tool" in prompt.lower()
        assert "educational" in prompt.lower()
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_result_construction(self, mock_anthropic):
        """Test that tool execution results are properly constructed"""
        # Create a more realistic mock response
        mock_initial_response = Mock()
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"  # This should be a string, not Mock
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        
        mock_initial_response.content = [mock_tool_block]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator(self.api_key, self.model)
        
        # Test the internal tool execution handler (now just returns tool results)
        tool_results = generator._handle_tool_execution(mock_initial_response, mock_tool_manager)
        
        # Verify tool results structure
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool result"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test"
        )
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic, sample_tool_definitions, 
                                               mock_sequential_tool_responses):
        """Test sequential tool calling across two rounds"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = mock_sequential_tool_responses
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager that returns different results for each call
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "ML is about algorithms learning from data",
            "Neural networks are inspired by the human brain"
        ]
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Compare machine learning and neural networks",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tool calls were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning basics")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="neural networks")
        
        # Verify three API calls were made (round 1, round 2, final)
        assert mock_client.messages.create.call_count == 3
        
        # Verify final result
        assert result == "Based on my searches, machine learning and neural networks are..."
        
        # Verify message structure progression
        call_args_list = mock_client.messages.create.call_args_list
        
        # First call: initial query
        first_call_messages = call_args_list[0][1]["messages"]
        assert len(first_call_messages) == 1
        assert first_call_messages[0]["role"] == "user"
        
        # Second call: includes first round's tool results
        second_call_messages = call_args_list[1][1]["messages"]
        assert len(second_call_messages) == 3  # user query + assistant tool use + user tool results
        
        # Third call: includes both rounds of conversation
        third_call_messages = call_args_list[2][1]["messages"]
        assert len(third_call_messages) == 5  # all previous messages + second round
    
    @patch('ai_generator.anthropic.Anthropic') 
    def test_sequential_tool_calling_early_termination(self, mock_anthropic, sample_tool_definitions,
                                                      mock_early_termination_responses):
        """Test early termination when Claude doesn't request second round"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = mock_early_termination_responses
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a high-level programming language"
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "What is Python?",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify only one tool call was executed
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="python basics")
        
        # Verify only two API calls were made (round 1 with tools, round 2 final response)
        assert mock_client.messages.create.call_count == 2
        
        # Verify final result
        assert result == "Python is a programming language..."
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_reached(self, mock_anthropic, sample_tool_definitions):
        """Test behavior when maximum rounds (2) is reached"""
        # Mock responses: both rounds request tools, then final response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_tool_block = Mock()
        round1_tool_block.type = "tool_use"
        round1_tool_block.name = "search_course_content"
        round1_tool_block.id = "tool_1"
        round1_tool_block.input = {"query": "first search"}
        round1_response.content = [round1_tool_block]
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_tool_block = Mock()
        round2_tool_block.type = "tool_use"
        round2_tool_block.name = "search_course_content"
        round2_tool_block.id = "tool_2"
        round2_tool_block.input = {"query": "second search"}
        round2_response.content = [round2_tool_block]
        
        # Final response after max rounds
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Final synthesized response")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["First result", "Second result"]
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Complex query requiring multiple searches",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tool calls were executed (max rounds reached)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify three API calls (2 rounds + final)
        assert mock_client.messages.create.call_count == 3
        
        # Verify final result
        assert result == "Final synthesized response"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_error_handling(self, mock_anthropic, sample_tool_definitions):
        """Test error handling in multi-round tool calling"""
        # Round 1: successful tool call
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_tool_block = Mock()
        round1_tool_block.type = "tool_use"
        round1_tool_block.name = "search_course_content"
        round1_tool_block.id = "tool_1"
        round1_tool_block.input = {"query": "test query"}
        round1_response.content = [round1_tool_block]
        
        # Round 2: API error
        round2_error = Exception("API rate limit exceeded")
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [round1_response, round2_error]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Successful first search"
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Query that will encounter an error",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify first tool was executed successfully
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify error was handled gracefully
        assert "Error:" in result
        assert "API rate limit exceeded" in result
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_context_preservation(self, mock_anthropic, sample_tool_definitions):
        """Test that conversation context is preserved across rounds"""
        # Round 1: tool call
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_tool_block = Mock()
        round1_tool_block.type = "tool_use"
        round1_tool_block.name = "search_course_content"
        round1_tool_block.id = "tool_1"
        round1_tool_block.input = {"query": "context test"}
        round1_response.content = [round1_tool_block]
        
        # Round 2: final response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        round2_response.content = [Mock(text="Response with preserved context")]
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [round1_response, round2_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        generator = AIGenerator(self.api_key, self.model)
        result = generator.generate_response(
            "Test conversation context",
            conversation_history="Previous conversation about AI topics",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify conversation history was included in both API calls
        call_args_list = mock_client.messages.create.call_args_list
        
        for call_args in call_args_list:
            system_content = call_args[1]["system"]
            assert "Previous conversation about AI topics" in system_content
        
        assert result == "Response with preserved context"