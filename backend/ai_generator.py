import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ConversationState:
    """Tracks conversation state across multiple rounds of tool calling"""
    messages: List[Dict[str, Any]]
    round_count: int = 0
    max_rounds: int = 2
    tools: Optional[List] = None
    tool_manager: Optional[Any] = None

@dataclass
class RoundResponse:
    """Encapsulates the result of a single round of Claude interaction"""
    claude_response: Any
    has_tool_calls: bool
    tool_results: List[Dict[str, Any]]
    final_text: Optional[str] = None
    error: Optional[str] = None

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or when users ask for course outlines
- **Multi-round tool usage**: You can make tool calls across up to 2 rounds to gather comprehensive information
- **Round strategy**: Consider what information you need and plan tool usage accordingly - search broadly first, then refine based on results if needed
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol for Course Outlines:
- When users ask for course outline, structure, or lesson list, use the course outline tool
- Always include the complete information returned: course title, course link, and all lessons with their numbers and titles
- Present the tool results exactly as returned without additional formatting or modification
- The tool already provides properly formatted output with bullet points and lesson structure

Response Protocol for Content Questions:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with multi-round tool usage support.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Initialize conversation state
        state = ConversationState(
            messages=[{"role": "user", "content": query}],
            max_rounds=2,  # Could be made configurable
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Execute rounds until termination
        for round_num in range(1, state.max_rounds + 1):
            state.round_count = round_num
            
            # Execute single round
            round_response = self._execute_single_round(state, system_content)
            
            # Check termination conditions
            if self._should_terminate(round_response, round_num, state.max_rounds):
                if round_response.final_text:
                    return round_response.final_text
                elif round_response.error:
                    return f"Error: {round_response.error}"
                else:
                    return "No response generated"
            
            # Prepare for next round
            self._prepare_next_round(state, round_response)
        
        # If we've reached max rounds without natural termination, 
        # make final call to get response
        return self._get_final_response(state, system_content)
    
    def _handle_tool_execution(self, claude_response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute tool calls and return results for next round.
        
        Args:
            claude_response: The response containing tool use requests
            tool_manager: Manager to execute tools
            
        Returns:
            List of tool result dictionaries
        """
        tool_results = []
        for content_block in claude_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        return tool_results
    
    def _execute_single_round(self, state: ConversationState, system_content: str) -> RoundResponse:
        """
        Execute a single round of Claude interaction.
        
        Args:
            state: Current conversation state
            system_content: System prompt content
            
        Returns:
            RoundResponse with results of this round
        """
        try:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": state.messages.copy(),
                "system": system_content
            }
            
            # Add tools if available
            if state.tools:
                api_params["tools"] = state.tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Call Claude
            claude_response = self.client.messages.create(**api_params)
            
            # Check if Claude wants to use tools
            if claude_response.stop_reason == "tool_use" and state.tool_manager:
                tool_results = self._handle_tool_execution(claude_response, state.tool_manager)
                return RoundResponse(
                    claude_response=claude_response,
                    has_tool_calls=True,
                    tool_results=tool_results
                )
            else:
                # No tool calls - final response
                return RoundResponse(
                    claude_response=claude_response,
                    has_tool_calls=False,
                    tool_results=[],
                    final_text=claude_response.content[0].text
                )
                
        except Exception as e:
            return RoundResponse(
                claude_response=None,
                has_tool_calls=False,
                tool_results=[],
                error=str(e)
            )
    
    def _should_terminate(self, round_response: RoundResponse, round_num: int, max_rounds: int) -> bool:
        """
        Determine if execution should terminate after current round.
        
        Args:
            round_response: Result of current round
            round_num: Current round number
            max_rounds: Maximum allowed rounds
            
        Returns:
            True if should terminate, False if should continue
        """
        # Terminate on error
        if round_response.error:
            return True
        
        # Terminate if Claude didn't request tools (natural completion)
        if not round_response.has_tool_calls:
            return True
        
        # Don't terminate just for max rounds - only if there are also no more tool calls needed
        # The final response will be generated outside the loop
        
        return False
    
    def _prepare_next_round(self, state: ConversationState, round_response: RoundResponse):
        """
        Update conversation state for next round.
        
        Args:
            state: Conversation state to update
            round_response: Results from completed round
        """
        # Add Claude's response with tool calls
        state.messages.append({
            "role": "assistant",
            "content": round_response.claude_response.content
        })
        
        # Add tool results
        if round_response.tool_results:
            state.messages.append({
                "role": "user",
                "content": round_response.tool_results
            })
    
    def _get_final_response(self, state: ConversationState, system_content: str) -> str:
        """
        Get final response after max rounds reached.
        
        Args:
            state: Current conversation state
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        try:
            # Make final API call without tools to get response
            api_params = {
                **self.base_params,
                "messages": state.messages.copy(),
                "system": system_content
            }
            
            final_response = self.client.messages.create(**api_params)
            return final_response.content[0].text
            
        except Exception as e:
            return f"Error generating final response: {str(e)}"