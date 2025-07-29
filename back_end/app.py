from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import logging
from datetime import datetime
import traceback
from therapist_agent import TherapistAgent
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

therapist_agent = None

def initialize_agent():
    """Initialize the therapist agent with error handling"""
    global therapist_agent
    try:
        therapist_agent = TherapistAgent(
            max_working_messages=10,
            summarize_threshold=20,
            max_reflection_attempts=2,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        logger.info("TherapistAgent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TherapistAgent: {str(e)}")
        return False

# Initialize on startup
if not initialize_agent():
    logger.error("Failed to start application - TherapistAgent initialization failed")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'agent_ready': therapist_agent is not None
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        # Check if agent is initialized
        if therapist_agent is None:
            return jsonify({
                'error': 'Therapist agent not initialized',
                'message': 'Service temporarily unavailable'
            }), 503

        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Please send JSON data with message field'
            }), 400

        # Extract message and session ID
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))

        # Validate message
        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'message': 'Message cannot be empty'
            }), 400

        # Log the request (be careful with sensitive data in production)
        logger.info(f"Processing message for session {session_id[:8]}...")

        # Get response from therapist agent
        response = therapist_agent.get_response(user_message, session_id)

        # Return successful response
        return jsonify({
            'response': response,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your message',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/new_session', methods=['POST'])
def new_session():
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        return jsonify({
            'session_id': session_id,
            'message': 'New session created',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error creating new session: {str(e)}")
        return jsonify({
            'error': 'Failed to create session',
            'message': str(e)
        }), 500

@app.route('/session/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a session (optional feature)"""
    try:
        if therapist_agent is None:
            return jsonify({
                'error': 'Therapist agent not initialized'
            }), 503

        # Get current state from the agent's memory
        thread_config = {"configurable": {"thread_id": session_id}}
        current_state = therapist_agent.graph.get_state(thread_config)
        
        if current_state.values and "messages" in current_state.values:
            # Format messages for frontend
            messages = []
            for msg in current_state.values["messages"]:
                role = "user" if hasattr(msg, 'content') and msg.__class__.__name__ == "HumanMessage" else "therapist"
                messages.append({
                    "role": role,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()  # Note: actual timestamps would need to be stored
                })
            
            return jsonify({
                'session_id': session_id,
                'messages': messages,
                'conversation_summary': current_state.values.get('conversation_summary', ''),
                'message_count': len(messages)
            })
        else:
            return jsonify({
                'session_id': session_id,
                'messages': [],
                'conversation_summary': '',
                'message_count': 0
            })

    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve session history',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The requested method is not allowed for this endpoint'
    }), 405

if __name__ == '__main__':
    # Development server configuration
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=True,      # Enable debug mode for development
        threaded=True    # Handle multiple requests concurrently
    )