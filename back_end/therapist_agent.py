from dotenv import load_dotenv

from typing import Dict, List, Any, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    conversation_summary: str
    crisis_detected: bool
    reflection_needed: bool
    reflection_attempts: int
    last_response_quality: str
    crisis_reflection_needed: bool
    crisis_reflection_attempts: int
    last_crisis_response_quality: str


class TherapistAgent:
    """A therapist chatbot with crisis detection and reflection capabilities."""
    
    def __init__(self, api_key: str, max_working_messages: int = 10, summarize_threshold: int = 20, max_reflection_attempts: int = 2):
        """Initialize the therapist bot."""
        # Set up the OpenAI model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.summary_llm = ChatOpenAI(
            model="mistralai/ministral-8b",
            temperature=0.3,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.crisis_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.reflection_llm = ChatOpenAI(
            model="mistralai/ministral-8b",
            temperature=0.2,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.crisis_reflection_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        self.memory = MemorySaver()
        self.max_working_messages = max_working_messages
        self.summarize_threshold = summarize_threshold
        self.max_reflection_attempts = max_reflection_attempts

        # Build the graph
        self.graph = self._build_graph()
    
    def _get_default_state(self) -> AgentState:
        """Return default state values."""
        return {
            "messages": [],
            "conversation_summary": "",
            "crisis_detected": False,
            "reflection_needed": False,
            "reflection_attempts": 0,
            "last_response_quality": "",
            "crisis_reflection_needed": False,
            "crisis_reflection_attempts": 0,  # Fixed: int instead of bool
            "last_crisis_response_quality": ""
        }
    
    def _build_graph(self) -> StateGraph:
        """Build a graph."""
        
        # Create the workflow
        workflow = StateGraph(AgentState)
        
        workflow.add_node("receive_message", self._receive_message)
        workflow.add_node("crisis_detection", self._crisis_detection)
        workflow.add_node("manage_memory", self._manage_memory)
        workflow.add_node("generate_response", self._generate_therapeutic_response)
        workflow.add_node("crisis_response", self._crisis_response)
        workflow.add_node("crisis_reflection", self._crisis_reflection)
        workflow.add_node("reflection", self._reflection)
        
        workflow.add_edge(START, "receive_message")
        workflow.add_edge("receive_message", "crisis_detection")
        
        workflow.add_conditional_edges(
            "crisis_detection",
            self._route_after_crisis_detection,
            {
                "crisis": "crisis_response",
                "normal": "manage_memory"
            }
        )
        
        # crisis branch
        workflow.add_edge("crisis_response", "crisis_reflection")
        workflow.add_conditional_edges(
            "crisis_reflection", 
            self._route_after_crisis_reflection,
            {"regenerate_crisis": "crisis_response", "end_crisis": END}
        )

        # normal branch
        workflow.add_edge("manage_memory", "generate_response")
        workflow.add_edge("generate_response", "reflection")
        workflow.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {
                "regenerate": "generate_response",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory)
    
    def _receive_message(self, state: AgentState) -> Dict[str, Any]:
        """First node: Receive message and ensure all state fields exist."""
        
        # Fixed: Return state updates instead of mutating directly
        updates = {}
        default_state = self._get_default_state()
        
        # Only add missing fields to updates
        for key, default_value in default_state.items():
            if key not in state or state[key] is None:
                updates[key] = default_value
        
        return updates
    
    def _crisis_detection(self, state: AgentState) -> Dict[str, Any]:
        """Detect if the user's message indicates a crisis situation."""
                
        crisis_prompt = f"""Analyze this message for crisis indicators. Look for:
        - Suicidal ideation or self-harm intentions
        - Threats of violence toward others
        - Immediate danger or emergency situations
        - Severe mental health crisis requiring immediate intervention
        
        Message: "{state["messages"][-1].content}"
        
        Respond with only "CRISIS" if immediate intervention is needed, or "SAFE" if not."""
            
        try:
            crisis_response = self.crisis_llm.invoke([HumanMessage(content=crisis_prompt)])
            crisis_detected = "CRISIS" in crisis_response.content.upper()
        except Exception as e:
            print(f"Error in crisis detection: {e}")
            crisis_detected = False
        
        if crisis_detected:
            print("⚠️ CRISIS DETECTED - Routing to crisis response")
        else:
            print("NO CRISIS DETECTED - Routing to default response")
        
        return {"crisis_detected": crisis_detected}
    
    def _route_after_crisis_detection(self, state: AgentState) -> Literal["crisis", "normal"]:
        """Route based on crisis detection."""
        return "crisis" if state["crisis_detected"] else "normal"
    
    def _crisis_response(self, state: AgentState) -> Dict[str, Any]:
        """Handle crisis situations with appropriate resources and response."""
        # this is where you could add tooling to trigger emergency hotlines
        
        crisis_system_prompt = """You are a crisis-trained therapist. The user may be in immediate danger.

        CRITICAL GUIDELINES:
        1. Express immediate concern and validation
        2. Ask directly about safety and immediate plans
        3. Provide crisis resources (988 Suicide & Crisis Lifeline, 911 for emergencies)
        4. Encourage professional help
        5. Stay calm but urgent
        6. Don't minimize their feelings
        7. Try to keep them talking

        Crisis Resources extremely crucial to mention:
        - 988 Suicide & Crisis Lifeline (call or text)
        - 911 for immediate emergencies
        - Crisis Text Line: Text HOME to 741741
        - National Domestic Violence Hotline: 1-800-799-7233
        
        Respond with empathy but prioritize safety."""

        if state.get("crisis_reflection_needed") and state.get("last_crisis_response_quality"):
            crisis_system_prompt += f"\n\nIMPROVEMENT NEEDED: {state['last_crisis_response_quality']}"
            crisis_system_prompt += "\nPlease address these concerns in your response while maintaining therapeutic principles."
        
        messages = [SystemMessage(content=crisis_system_prompt)] + state["messages"]
        
        try:
            response = self.crisis_llm.invoke(messages)
            new_messages = state["messages"] + [response]
            
            # Log crisis intervention
            print(f"🚨 CRISIS INTERVENTION ACTIVATED")
            print(f"User message: {state['messages'][-1].content[:100]}...")
            print(f"Crisis response provided")
            
            return {"messages": new_messages}
            
        except Exception as e:
            print(f"Error in crisis response: {e}")
            # Fallback crisis response
            fallback_response = """I'm very concerned about what you're sharing. Your safety is the most important thing right now. 

            Please reach out for immediate help:
            • Call or text 988 (Suicide & Crisis Lifeline) - available 24/7
            • Call 911 if you're in immediate danger
            • Go to your nearest emergency room
            
            You don't have to go through this alone. There are people who want to help you."""
            
            new_messages = state["messages"] + [AIMessage(content=fallback_response)]
            return {"messages": new_messages}
    
    def _crisis_reflection(self, state: AgentState) -> Dict[str, Any]:
        """Reflect on the quality of the crisis response and decide if improvement is needed."""
        
        # Check if we've exceeded max reflection attempts
        if state["crisis_reflection_attempts"] >= self.max_reflection_attempts:
            print(f"🔄 Max crisis reflection attempts ({self.max_reflection_attempts}) reached. Accepting current response.")
            return {"crisis_reflection_needed": False}
        
        # Get the last exchange (user message and therapist response)
        if len(state["messages"]) >= 2:
            user_msg = state["messages"][-2].content
            therapist_msg = state["messages"][-1].content
        
            crisis_reflection_prompt = f"""You are a STRICT crisis intervention supervisor evaluating emergency therapeutic responses. Use CRISIS-SPECIFIC standards and be critically thorough.

            User said: "{user_msg}"
            Crisis Therapist responded: "{therapist_msg}"
            
            CRISIS RESPONSE SCORING SYSTEM - Score each criterion 1-5 (5 = exceptional, 4 = good, 3 = acceptable, 2 = poor, 1 = unacceptable):

            1. IMMEDIATE SAFETY ASSESSMENT (Must directly address safety): _/5
            - Directly asks about immediate safety/danger
            - Assesses current location and support
            - Inquires about specific plans or means
            - Shows urgency appropriate to crisis level

            2. CRISIS RESOURCE PROVISION (Must provide specific resources): _/5
            - Includes 988 Suicide & Crisis Lifeline
            - Mentions 911 for immediate emergencies
            - Provides Crisis Text Line (HOME to 741741)
            - Includes other relevant hotlines when appropriate
            - Resources are prominently displayed and actionable

            3. VALIDATION & DE-ESCALATION (Must acknowledge crisis emotions): _/5
            - Validates the person's pain without minimizing
            - Shows genuine concern and empathy
            - Avoids judgment or dismissive language
            - Communicates that help is available
            - Expresses that they matter and are not alone

            4. PROFESSIONAL CRISIS RESPONSE (Must maintain crisis standards): _/5
            - Uses calm but urgent tone
            - Maintains professional boundaries in crisis
            - Demonstrates crisis training competence
            - Avoids casual language inappropriate for crisis
            - Shows appropriate level of concern

            5. ENGAGEMENT & SAFETY PLANNING (Must encourage continued contact): _/5
            - Encourages staying on the line/continuing conversation
            - Asks about support systems or safe people
            - Suggests immediate next steps for safety
            - Creates connection and reduces isolation
            - Focuses on short-term safety planning

            CRISIS-SPECIFIC SCORING GUIDELINES:
            - 5: Exceptional crisis response, could save a life
            - 4: Good crisis response with minor areas for enhancement
            - 3: Acceptable crisis response but lacks some key elements
            - 2: Poor crisis response - missing critical safety elements
            - 1: Unacceptable - dangerous or inappropriate for crisis

            AUTOMATIC SCORE DEDUCTIONS FOR CRISIS RESPONSES:
            - Missing 988 hotline: -3 points from Resources
            - No direct safety assessment: -4 points from Safety Assessment
            - Minimizing crisis ("it's not that bad"): -3 points from Validation
            - Generic therapeutic response (treating like regular session): -2 points all categories
            - No urgency or immediate action steps: -2 points from Engagement
            - Missing "you're not alone" messaging: -2 points from Validation

            MINIMUM PASSING SCORE FOR CRISIS: 22/25 (88%) - Higher standard due to life-threatening nature
            
            Provide your scoring like this:
            - Safety Assessment: X/5 (brief explanation)
            - Crisis Resources: X/5 (brief explanation)  
            - Validation & De-escalation: X/5 (brief explanation)
            - Professional Crisis Response: X/5 (brief explanation)
            - Engagement & Safety Planning: X/5 (brief explanation)

            TOTAL CRISIS SCORE: X/25

            If score is 22+ respond: "CRISIS_GOOD - Score: X/25"
            If score is under 22 respond: "CRISIS_IMPROVE: Score X/25 - [detailed feedback focusing on safety-critical improvements needed]"

            Remember: In crisis situations, lives may depend on the quality of this response. Be especially strict about safety elements."""
            
            try:
                reflection_response = self.crisis_reflection_llm.invoke([HumanMessage(content=crisis_reflection_prompt)])
                evaluation = reflection_response.content.strip()
                
                import re
                score_match = re.search(r'Score[:\s]*(\d+)/25', evaluation)
                if score_match:
                    score = int(score_match.group(1))
                    print(f"DEBUG: Extracted crisis score: {score}/25")
                    
                    if score < 22:  # Higher threshold for crisis responses (88%)
                        print(f"🚨 CRISIS REFLECTION: Response needs improvement (Score: {score}/25, Attempt {state['crisis_reflection_attempts'] + 1}/{self.max_reflection_attempts})")
                        print(f"Critical Safety Feedback: {evaluation}")
                        # Remove the last response so it can be regenerated
                        new_messages = state["messages"][:-1]
                        return {
                            "crisis_reflection_needed": True,
                            "crisis_reflection_attempts": state["crisis_reflection_attempts"] + 1,
                            "last_crisis_response_quality": evaluation,
                            "messages": new_messages
                        }
                    else:
                        print(f"✅ CRISIS REFLECTION: Crisis response approved - Score: {score}/25")
                        return {
                            "crisis_reflection_needed": False,
                            "last_crisis_response_quality": evaluation
                        }
                else:
                    # Fallback to string-based check with crisis-specific keywords
                    if "CRISIS_IMPROVE" in evaluation.upper():
                        print(f"🚨 CRISIS REFLECTION: Response needs improvement (Attempt {state['crisis_reflection_attempts'] + 1}/{self.max_reflection_attempts})")
                        print(f"Critical Safety Feedback: {evaluation}")
                        new_messages = state["messages"][:-1]
                        return {
                            "crisis_reflection_needed": True,
                            "crisis_reflection_attempts": state["crisis_reflection_attempts"] + 1,
                            "last_crisis_response_quality": evaluation,
                            "messages": new_messages
                        }
                    else:
                        print(f"✅ CRISIS REFLECTION: Crisis response approved - {evaluation}")
                        return {
                            "crisis_reflection_needed": False,
                            "last_crisis_response_quality": evaluation
                        }
                    
            except Exception as e:
                print(f"Error in crisis reflection: {e}")
                return {"crisis_reflection_needed": False}
        else:
            return {"crisis_reflection_needed": False}
    
    def _route_after_crisis_reflection(self, state: AgentState) -> Literal["regenerate_crisis", "end_crisis"]:
        """Route based on crisis reflection results."""
        return "regenerate_crisis" if state["crisis_reflection_needed"] else "end_crisis"
    
    def _manage_memory(self, state: AgentState) -> Dict[str, Any]:
        """Second node: Manage conversation memory and truncation."""

        # Check if we need to truncate
        if len(state["messages"]) > self.summarize_threshold:
            print(f"Truncating messages. Total: {len(state['messages'])}")
            
            # Keep the most recent messages
            recent_messages = state["messages"][-self.max_working_messages:]
            
            # Messages to summarize (everything except recent ones)
            messages_to_summarize = state["messages"][:-self.max_working_messages]
            
            # Generate or update summary
            if messages_to_summarize:
                new_summary = self._create_conversation_summary(
                    messages_to_summarize, 
                    state.get("conversation_summary", "")
                )
                
                print(f"Truncated to {len(recent_messages)} working messages")
                return {
                    "conversation_summary": new_summary,
                    "messages": recent_messages
                }
        
        return {}  # No updates needed
    
    def _create_conversation_summary(self, messages_to_summarize: List[BaseMessage], 
                                   existing_summary: str = "") -> str:
        """Create a summary of messages being truncated."""
        
        # Format messages for summarization
        conversation_text = ""
        for msg in messages_to_summarize:
            role = "User" if isinstance(msg, HumanMessage) else "Therapist"
            conversation_text += f"{role}: {msg.content}\n"
        
        # Create summarization prompt
        summary_prompt = f"""You are summarizing a therapy conversation for continuity. 
        Create a concise summary that captures:
        - Key topics discussed
        - Important emotional themes
        - Any breakthroughs or significant moments
        - Ongoing concerns or patterns
        - Any crisis situations or safety concerns (mark as HIGH PRIORITY)
        
        Existing summary: {existing_summary}
        
        New conversation to add to summary:
        {conversation_text}
        
        Provide an updated, comprehensive summary:"""
        
        try:
            summary_response = self.summary_llm.invoke([HumanMessage(content=summary_prompt)])
            return summary_response.content
        except Exception as e:
            print(f"Error creating summary: {e}")
            return existing_summary or "Previous conversation context available but summary unavailable."
    
    def _generate_therapeutic_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate a therapeutic response using the LLM."""
        
        system_prompt = """You are a compassionate, skilled therapist. Use evidence-based therapeutic techniques:

        CORE PRINCIPLES:
        - Active listening and reflection
        - Validation of emotions
        - Open-ended questions to promote insight
        - Cognitive-behavioral techniques when appropriate
        - Mindfulness and grounding techniques
        - Strength-based approach
        
        TECHNIQUES TO USE:
        - Reflection: "It sounds like you're feeling..."
        - Clarification: "Help me understand..."
        - Reframing: "Another way to look at this might be..."
        - Scaling: "On a scale of 1-10, how intense is this feeling?"
        - Homework/exercises: Suggest practical coping strategies
        
        Always be warm, non-judgmental, and professional. Avoid giving direct advice - instead, help the user discover their own insights."""
        
        # Add improvement feedback if this is a regeneration
        if state.get("reflection_needed") and state.get("last_response_quality"):
            system_prompt += f"\n\nIMPROVEMENT NEEDED: {state['last_response_quality']}"
            system_prompt += "\nPlease address these concerns in your response while maintaining therapeutic principles."
        
        if state["conversation_summary"]:
            system_prompt += f"\n\nConversation Context: {state['conversation_summary']}"

        # Variable that includes SystemMessage to message history for the LLM
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Get response from OpenAI
        try:
            response = self.llm.invoke(messages)
            new_messages = state["messages"] + [response]
            print("conversation summary: ", state["conversation_summary"])
            return {"messages": new_messages}
        except Exception as e:
            print(f"Error generating therapeutic response: {e}")
            fallback_response = "I apologize, but I'm experiencing some technical difficulties. Could you please share what's on your mind again? I'm here to listen and support you."
            new_messages = state["messages"] + [AIMessage(content=fallback_response)]
            return {"messages": new_messages}
    
    def _reflection(self, state: AgentState) -> Dict[str, Any]:
        """Reflect on the quality of the therapeutic response and decide if improvement is needed."""
        
        # Check if we've exceeded max reflection attempts
        if state["reflection_attempts"] >= self.max_reflection_attempts:
            print(f"🔄 Max reflection attempts ({self.max_reflection_attempts}) reached. Accepting current response.")
            return {"reflection_needed": False}
        
        # Get the last exchange (user message and therapist response)
        if len(state["messages"]) >= 2:
            user_msg = state["messages"][-2].content
            therapist_msg = state["messages"][-1].content
        
            reflection_prompt = f"""You are a STRICT clinical supervisor evaluating therapeutic responses. Use the HIGHEST professional standards and be critically thorough.

            User said: "{user_msg}"
            Therapist responded: "{therapist_msg}"
            
            MULTI-CRITERIA SCORING SYSTEM - Score each criterion 1-5 (5 = exceptional, 4 = good, 3 = acceptable, 2 = poor, 1 = unacceptable):

            1. EMPATHY & VALIDATION (Must acknowledge emotions explicitly): _/5
               - Does the response show deep understanding?
               - Are emotions explicitly validated and normalized?
               - Is there genuine warmth and acceptance?

            2. THERAPEUTIC TECHNIQUE (Must use specific evidence-based methods): _/5
               - Reflective listening ("It sounds like...", "I hear that...")
               - Open-ended questions that promote insight
               - Reframing or offering new perspectives
               - CBT, mindfulness, or other therapeutic techniques
               - NOT giving direct advice

            3. DEPTH & INSIGHT PROMOTION (Must go beyond surface level): _/5
               - Helps user explore underlying thoughts/feelings
               - Connects patterns or themes
               - Encourages self-discovery rather than providing answers
               - Addresses root causes, not just symptoms

            4. PROFESSIONAL LANGUAGE & BOUNDARIES (Must sound like trained therapist): _/5
               - Uses appropriate therapeutic language
               - Maintains professional boundaries
               - Avoids casual/friendship tone
               - Demonstrates clinical competence

            5. ENGAGEMENT & THERAPEUTIC ALLIANCE (Must invite deeper exploration): _/5
               - Asks specific, thought-provoking questions
               - Invites continued dialogue
               - Shows investment in client's growth
               - Creates safe space for vulnerability

            SCORING GUIDELINES:
            - 5: Exceptional therapeutic response, textbook quality
            - 4: Good therapeutic response with minor areas for enhancement
            - 3: Acceptable but lacks depth or technique
            - 2: Poor - missing key therapeutic elements
            - 1: Unacceptable - not therapeutic in nature

            AUTOMATIC SCORE DEDUCTIONS:
            - Generic/template responses: -2 points per criterion
            - Direct advice-giving: -2 points from Technique
            - No emotional validation: -3 points from Empathy
            - No questions asked: -2 points from Engagement
            - Too brief (under 3 sentences): -1 point overall
            - Casual tone: -2 points from Professional Language

            MINIMUM PASSING SCORE: 18/25
            
            Provide your scoring like this:
            - Empathy & Validation: X/5 (brief explanation)
            - Therapeutic Technique: X/5 (brief explanation)  
            - Depth & Insight: X/5 (brief explanation)
            - Professional Language: X/5 (brief explanation)
            - Engagement & Questions: X/5 (brief explanation)

            TOTAL SCORE: X/25

            If score is 20+ respond: "GOOD - Score: X/25"
            If score is under 20 respond: "IMPROVE: Score X/25 - [detailed feedback on what needs improvement]"

            Be ruthlessly honest in your evaluation - therapeutic quality depends on it."""
            
            try:
                reflection_response = self.reflection_llm.invoke([HumanMessage(content=reflection_prompt)])
                evaluation = reflection_response.content.strip()
                
                import re
                score_match = re.search(r'Score[:\s]*(\d+)/25', evaluation)
                if score_match:
                    score = int(score_match.group(1))
                    print(f"DEBUG: Extracted score: {score}/25")
                    
                    if score < 20:  # Below passing threshold
                        print(f"🔄 STRICT REFLECTION: Response needs improvement (Score: {score}/25, Attempt {state['reflection_attempts'] + 1}/{self.max_reflection_attempts})")
                        print(f"Detailed Feedback: {evaluation}")
                        # Remove the last response so it can be regenerated
                        new_messages = state["messages"][:-1]
                        return {
                            "reflection_needed": True,
                            "reflection_attempts": state["reflection_attempts"] + 1,
                            "last_response_quality": evaluation,
                            "messages": new_messages
                        }
                    else:
                        print(f"✅ STRICT REFLECTION: Response quality approved - Score: {score}/25")
                        return {
                            "reflection_needed": False,
                            "last_response_quality": evaluation
                        }
                else:
                    # Fallback to original string-based check
                    if "IMPROVE" in evaluation.upper():
                        print(f"🔄 STRICT REFLECTION: Response needs improvement (Attempt {state['reflection_attempts'] + 1}/{self.max_reflection_attempts})")
                        print(f"Detailed Feedback: {evaluation}")
                        new_messages = state["messages"][:-1]
                        return {
                            "reflection_needed": True,
                            "reflection_attempts": state["reflection_attempts"] + 1,
                            "last_response_quality": evaluation,
                            "messages": new_messages
                        }
                    else:
                        print(f"✅ STRICT REFLECTION: Response quality approved - {evaluation}")
                        return {
                            "reflection_needed": False,
                            "last_response_quality": evaluation
                        }
                    
            except Exception as e:
                print(f"Error in reflection: {e}")
                return {"reflection_needed": False}
        else:
            return {"reflection_needed": False}
    
    def _route_after_reflection(self, state: AgentState) -> Literal["regenerate", "end"]:
        """Route based on reflection results."""
        return "regenerate" if state["reflection_needed"] else "end"
    
    def get_response(self, user_message: str, session_id: str = "default") -> str:
        """Main method to get a response from the bot."""
        
        # Need session_id for memory to work
        thread_config = {"configurable": {"thread_id": session_id}}
        
        # Get current state or create new one
        current_state = self.graph.get_state(thread_config)
        
        # Handle conversation history properly
        if current_state.values and "messages" in current_state.values:
            # Continue existing conversation
            state = current_state.values.copy()  # Make a copy to avoid mutation
            state["messages"] = state["messages"] + [HumanMessage(content=user_message)]
            # Fixed: Reset both reflection attempts for new user message
            state["reflection_attempts"] = 0
            state["crisis_reflection_attempts"] = 0  # Fixed: Reset crisis attempts too
        else:
            # Start new conversation with default state
            state = self._get_default_state()
            state["messages"] = [HumanMessage(content=user_message)]
        
        # Run through our graph with memory
        final_state = self.graph.invoke(state, thread_config)
        
        # Debug output
        if final_state.get("crisis_detected"):
            print("⚠️ Crisis intervention was activated for this response")
        if final_state.get("last_response_quality"):
            quality_preview = final_state['last_response_quality'][:80] + "..." if len(final_state['last_response_quality']) > 80 else final_state['last_response_quality']
            print(f"🔍 Final response quality: {quality_preview}")
        
        # Get the latest AI response from conversation history
        latest_response = final_state["messages"][-1].content
        return latest_response
    
    # Helper functions
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("therapy_workflow.png", "wb") as f:
                f.write(png_data)
            
            print("Workflow saved as therapy_workflow.png")
            
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = TherapistAgent(max_working_messages=10, summarize_threshold=20, max_reflection_attempts=2, api_key=os.getenv("OPENROUTER_API_KEY"))
    
    print("Enhanced Therapist Agent with Crisis Detection & Reflection ready!")
    print("Features: Crisis intervention, Response quality reflection, Memory management")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        response = agent.get_response(user_input)
        print(f"Therapist: {response}\n")