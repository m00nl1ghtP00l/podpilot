# Conversational Learning Agent

## Overview

The conversational agent provides an interactive chat interface using Ollama, allowing you to have a back-and-forth conversation to create learning packs.

## Key Features

### 1. **Conversation History**
- Maintains context across multiple messages
- Remembers what you've discussed
- Can reference previous parts of the conversation

### 2. **Intent Understanding**
- Uses LLM to understand what you want to do
- Can handle:
  - Creating learning packs
  - Asking questions
  - Modifying requests
  - Getting suggestions
  - Checking status

### 3. **Natural Language Interaction**
- Chat naturally instead of using command-line flags
- Ask follow-up questions
- Get clarifications when needed

## Usage

### Basic Usage

```bash
python conversational_agent.py
```

### With Custom Model

```bash
python conversational_agent.py --model qwen2.5:14b
```

## Example Conversation

```
You: I want to learn business Japanese vocabulary

Assistant: I can help you create a learning pack on business Japanese vocabulary! 
Do you have any specific channels or episodes in mind, or should I search through 
all available lessons?

You: Use the hnh channel, focus on recent episodes

Assistant: Found 5 episodes matching your criteria in the 'hnh' channel.
Would you like me to:
- Download and transcribe these episodes?
- Use existing lessons if available?
- Create a learning pack right away?

You: Create a pack from existing lessons

Assistant: ✓ Created learning pack with 3 lessons!
Saved to: learning_pack_20241209_143022.md

Would you like me to:
- Create another pack on a different topic?
- Find more episodes?
- Answer questions about the content?

You: What topics are covered in those lessons?

Assistant: The lessons cover:
- Business greetings and introductions
- Office communication phrases
- Meeting vocabulary
- Email expressions

Would you like me to create a more focused pack on any of these topics?
```

## How It Works

### 1. **Conversation Loop**
```python
while True:
    user_input = input("You: ")
    response = agent.chat(user_input)
    print(f"Assistant: {response}")
```

### 2. **Intent Classification**
The agent uses LLM to understand your intent:
- `create_pack` - Create a learning pack
- `ask_question` - Answer questions
- `modify_request` - Change previous request
- `get_suggestions` - Get learning suggestions
- `check_status` - Check current status

### 3. **Context Management**
```python
# Maintains conversation history
conversation_history = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

### 4. **Tool Execution**
When you ask to create a pack, the agent:
1. Understands your request
2. Finds relevant lessons
3. Generates the pack
4. Saves it to a file
5. Offers next steps

## Differences from `learning_agent.py`

| Feature | `learning_agent.py` | `conversational_agent.py` |
|---------|---------------------|---------------------------|
| Interaction | One-shot command | Interactive chat |
| History | None | Full conversation history |
| Follow-ups | No | Yes, can ask follow-up questions |
| Clarifications | No | Yes, asks for clarification |
| Context | Single query | Full conversation context |

## Requirements

- Ollama running locally (`ollama serve`)
- Model installed (e.g., `ollama pull qwen2.5:14b`)
- Python dependencies: `requests`

## Tips

1. **Be specific**: "I want to learn business Japanese from hnh channel" is better than "I want to learn Japanese"

2. **Ask follow-ups**: After creating a pack, ask questions like:
   - "What topics are covered?"
   - "Can you create another pack on X?"
   - "Find more episodes on Y"

3. **Use natural language**: Talk to it like you would a human assistant

4. **Check status**: Ask "What's my current status?" to see what you're working with

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check the base URL matches your Ollama setup

### "Model not found"
- Install the model: `ollama pull qwen2.5:14b`
- Or use a different model with `--model`

### "No lessons available"
- Generate some lessons first using `generate_lesson.py`
- Or ask the agent to find and generate episodes

