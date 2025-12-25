import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import logger


class Memory:
    def __init__(self):
        self.sessions = {}
        logger.info("Memory initialized")

    def get_history(self, session_id):
        history = self.sessions.get(session_id, [])
        logger.debug(f"Retrieved history for session {session_id}: {len(history)} messages")
        return history

    def add_message(self, session_id, role, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"New session created: {session_id}")
        
        self.sessions[session_id].append({"role": role, "content": content})
        
        msg_preview = content[:100] + "..." if len(content) > 100 else content
        logger.info(f"[{session_id}] {role.upper()}: {msg_preview}")
        logger.debug(f"Session {session_id} now has {len(self.sessions[session_id])} messages")
    
    def get_session_count(self):
        return len(self.sessions)
    
    def clear_session(self, session_id):
        if session_id in self.sessions:
            msg_count = len(self.sessions[session_id])
            del self.sessions[session_id]
            logger.info(f"Session {session_id} cleared ({msg_count} messages)")
        else:
            logger.warning(f"Session {session_id} not found")