# OpenWebUI - Anthropic, DeepSeek & Cost Tracking Integrations

This repository provides three powerful integrations for OpenWebUI:
1. **Anthropic Claude Models** integration
2. **DeepSeek Models** integration 
3. **Cost Tracker** for monitoring token usage and costs

## Quick Setup

You can easily add these integrations to OpenWebUI by visiting the following links and pressing the **Get** button:

- **Anthropic Integration**: [Get Anthropic](https://openwebui.com/f/pescheckbram/deepseek_intergration_chat)
- **Deepseek Integration**: [Get Deepseek](https://openwebui.com/f/pescheckbram/anthropic_integration_chat)
- **Cost Tracker**: [Get Cost Filter](https://openwebui.com/f/pescheckbram/live_cost_tracker_when_chatting)

For all your submissions and updates, visit:
- **Open WebUI Main Page**: [OpenWebUI Submissions](https://openwebui.com/u/pescheckbram)
- **GitHub for Pipelines & Contributions**: [GitHub Repository](https://github.com/brammittendorff/openwebui-pipelines)

---

## Anthropic Integration

The Anthropic integration enables you to use Claude models directly within OpenWebUI.

### Features
- Support for the full Claude model lineup
- Handles text and multimodal inputs
- Stream response support for real-time replies

### Available Models
- `claude-3-haiku`
- `claude-3-opus`
- `claude-3-sonnet`
- `claude-3.5-haiku`
- `claude-3.5-sonnet`
- `claude-3.7-sonnet`

### Setup
1. Install the integration from the link above
2. Configure your Anthropic API key in the OpenWebUI admin panel
3. Select a Claude model from the model dropdown in the chat interface

---

## DeepSeek Integration

The DeepSeek integration brings DeepSeek's language models to OpenWebUI.

### Features
- Access to DeepSeek's conversational models
- Support for advanced reasoning capabilities
- Stream response support

### Available Models
- `deepseek-chat`
- `deepseek-reasoner`

### Setup
1. Install the integration from the link above
2. Configure your DeepSeek API key in the OpenWebUI admin panel
3. Select a DeepSeek model from the model dropdown in the chat interface

---

## Cost Tracker

The Cost Tracker is a filter that provides real-time cost monitoring for all your AI model interactions.

### Features
- **Live Cost Calculation**: See the cost of each conversation immediately
- **Token Counting**: Accurately counts input and output tokens
- **Performance Metrics**: Shows elapsed time and tokens/second processing speed
- **Persistent Storage**: Maintains a database of all interactions for cost auditing
- **Multi-Model Support**: Works with all models including Anthropic, OpenAI, DeepSeek, and others
- **Up-to-date Pricing**: Automatically fetches latest pricing from LiteLLM's repository

### How It Works
The Cost Tracker works transparently in the background to:
1. Count tokens in your prompts
2. Track processing time
3. Count tokens in the AI's responses
4. Calculate costs based on current model pricing
5. Display a statistics line showing time, speed, tokens, and cost

### Example Output
After each conversation, you'll see statistics like:
```
1.53 s | 45.76 T/s | 70 Tokens | $0.000350
```

### Setup
1. Install the Cost Tracker filter from the link above
2. No additional configuration is required - it works automatically!
3. Costs are logged to a database for future reference and tracking

---

## Troubleshooting

### 1. API Key Issues
- Ensure that API keys are correctly set in the Admin UI
- Check that API keys have proper permissions and are active

### 2. Logs & Errors
- Check logs for errors in OpenWebUI's log section
- For the Cost Tracker, enable debug mode in its settings for more detailed logs

### 3. Connection Issues
- Ensure your server has internet access
- Verify your server can reach the required API endpoints:
  - Anthropic: `api.anthropic.com`
  - DeepSeek: `api.deepseek.com`
  - LiteLLM (for pricing data): `raw.githubusercontent.com`

---

## License
This project is licensed under the **MIT License**.

For support, open an issue on GitHub: [OpenWebUI Pipelines](https://github.com/brammittendorff/openwebui-pipelines).