# Visual Novel Text Database

This database contains structured text files for use with the FAISS RAG system.

## File Structure

### Character Files (character_*.txt)
Contains personality, background, likes, dislikes, secrets, and dialogue style.
These help the AI understand how each character should respond.

### Location Files (location_*.txt)
Describes settings, atmosphere, and environmental details.
Provides context for scene-appropriate responses.

### World Lore (world_lore.txt)
Contains background information about the world, history, and setting.

### Dialogue Scenarios (dialogue_scenarios.jsonl)
Example dialogues for various situations in JSONL format.
Each line is a JSON object with scenario context and example text.

## Usage

1. Place all generated files in a folder (e.g., ./data)
2. Run: python rag_faiss_local.py ingest ./data
3. Start chatting: python rag_faiss_local.py chat

## Tips

- Keep character descriptions consistent
- Add specific details that make characters memorable
- Include varied dialogue examples for different moods/situations
- Update the database regularly as your story develops
