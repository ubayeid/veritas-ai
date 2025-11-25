"""
Main script to run the Compliance RAG Chatbot
"""

import sys
from pathlib import Path

# Add backend/searching to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from chatbot import Chatbot


def main():
    """Main entry point for the chatbot."""
    # Get base directory (project root)
    base_dir = Path(__file__).parent.parent.parent
    
    # Check for --hybrid flag
    use_hybrid = "--hybrid" in sys.argv
    
    print(f"Initializing chatbot with base directory: {base_dir}")
    
    try:
        chatbot = Chatbot(str(base_dir), use_hybrid=use_hybrid)
        chatbot.run_interactive()
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

