# Datasets Directory

This directory is for your custom datasets. The training system supports multiple formats.

## Supported Formats

### 1. JSONL (JSON Lines) - **Recommended for conversations**

One JSON object per line. Great for large datasets and streaming.

#### For Language Modeling (Conversations):
```jsonl
{"text": "User: How are you?\nAssistant: I'm doing well, thanks for asking!"}
{"text": "User: What's the weather like?\nAssistant: I don't have access to weather data."}
{"text": "User: Tell me a joke.\nAssistant: Why did the chicken cross the road? To get to the other side!"}
```

#### For Classification:
```jsonl
{"sentence": "This movie was amazing!", "label": 1}
{"sentence": "I hated this film.", "label": 0}
{"sentence": "It was okay, nothing special.", "label": 0}
```

### 2. JSON

Array of objects in a single file.

```json
[
  {"text": "User: Hello!\nAssistant: Hi there!"},
  {"text": "User: How can I help?\nAssistant: I'm here to assist you!"},
  {"text": "User: Thanks!\nAssistant: You're welcome!"}
]
```

### 3. Plain Text (.txt, .text)

One sample per line. Automatically used for language modeling.

```
User: What is AI?\nAssistant: AI stands for Artificial Intelligence.
User: Is it useful?\nAssistant: Yes, AI has many applications.
User: Give me an example.\nAssistant: AI can help with image recognition, language translation, and more.
```

## Conversation Formatting Tips

### Option 1: Simple format (works as-is)
```
User: How are you?\nAssistant: I'm doing well!
```

### Option 2: With special tokens (recommended)
```
<|user|>How are you?<|assistant|>I'm doing well!<|user|>What's your name?<|assistant|>I'm an AI assistant.
```

### Option 3: Multi-turn with turns separated
```
[User]: How are you?
[Assistant]: I'm doing well, thanks!
[User]: That's great to hear.
[Assistant]: Thank you for asking!
```

## File Naming

- Use descriptive names: `my_conversations.jsonl`, `customer_support_chats.json`
- The system will auto-detect format based on extension
- Multiple files can coexist - you'll select which one to use

## Data Split

- Automatically splits into 80% train / 20% validation
- Make sure you have enough data (minimum 1000 samples recommended)

## Example Datasets

Create a file like `example_conversations.jsonl`:

```jsonl
{"text": "User: Hello! How are you today?\nAssistant: I'm doing great, thank you for asking! How can I help you?"}
{"text": "User: Can you tell me about geometric attention?\nAssistant: Geometric attention uses learnable curvature to operate in different geometric spaces like hyperbolic and spherical manifolds."}
{"text": "User: That sounds interesting!\nAssistant: Yes! It shows a universal 50/50 split between hyperbolic and spherical heads across all NLP tasks."}
```

Then run `python main.py` and select "Local Dataset" to use it!

## Notes

- For conversations, the model learns to predict the next token
- No special preprocessing needed - just raw text
- The LanguageModelingDataset handles tokenization automatically
- Padding tokens are masked during training
