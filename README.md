# OpenWebUI Integrations

This repository provides three powerful integrations for [OpenWebUI](https://openwebui.com):

1. **Anthropic Claude Models Integration**: Access all Claude models directly in OpenWebUI
2. **DeepSeek Models Integration**: Use DeepSeek's powerful language models in OpenWebUI
3. **Cost Tracker Filter**: Real-time token usage and cost monitoring for all models

## Quick Installation

Install these integrations directly from OpenWebUI:

- **Anthropic Integration**: [Install Claude Models](https://openwebui.com/f/pescheckbram/anthropic_integration_chat_and_scan_image)
- **DeepSeek Integration**: [Install DeepSeek Models](https://openwebui.com/f/pescheckbram/deepseek_intergration_chat)
- **Cost Tracker**: [Install Cost Filter](https://openwebui.com/f/pescheckbram/live_cost_tracker_when_chatting)

Find all my submissions at [my OpenWebUI profile](https://openwebui.com/u/pescheckbram).

## Features Overview

### Anthropic Claude Integration (`pipes/anthropic_intergration.py`)

- Support for all Claude models (3, 3.5, and 3.7 series)
- Handles both text and image inputs (multimodal support)
- Real-time streaming responses
- Complete message history management

#### Supported Models

| Model | Description |
|-------|-------------|
| `claude-3-haiku` | Fast and cost-effective |
| `claude-3-opus` | Most capable Claude model |
| `claude-3-sonnet` | Balance of speed and capability |
| `claude-3.5-haiku` | Enhanced speed and efficiency |
| `claude-3.5-sonnet` | Improved reasoning at moderate cost |
| `claude-3.7-sonnet` | Latest model with advanced reasoning |

### DeepSeek Integration (`pipes/deepseek_intergration.py`)

- Access to DeepSeek's specialized language models
- Optimized for complex reasoning and technical content
- Real-time streaming responses
- Thoughtfully designed for OpenWebUI's chat interface

#### Supported Models

| Model | Description |
|-------|-------------|
| `deepseek-chat` | General purpose chat model |
| `deepseek-reasoner` | Enhanced reasoning capabilities |

### Cost Tracker Filter (`filters/cost_filter.py`)

- **Live Cost Monitoring**: See the cost of each conversation in real time
- **Token Counting**: Accurate input and output token counting for all models
- **Performance Metrics**: Track elapsed time and tokens per second
- **Universal Compatibility**: Works with all models (Anthropic, OpenAI, DeepSeek, etc.)
- **Up-to-date Pricing**: Fetches latest model pricing data automatically

#### Sample Output

After each conversation, you'll see statistics like:
```
1.53 s | 45.76 T/s | 70 Tokens | $0.000350
```

## Installation Instructions

### Prerequisites

- A working OpenWebUI installation
- API keys for the services you want to use:
  - Anthropic API key for Claude models
  - DeepSeek API key for DeepSeek models

### Manual Installation

If you prefer to install manually:

1. Clone this repository:
   ```bash
   git clone https://github.com/brammittendorff/openwebui-pipelines.git
   ```

2. Copy the desired integration files to your OpenWebUI installation:
   - For Anthropic: Copy `pipes/anthropic_intergration.py` to your OpenWebUI pipes directory
   - For DeepSeek: Copy `pipes/deepseek_intergration.py` to your OpenWebUI pipes directory
   - For Cost Tracker: Copy `filters/cost_filter.py` to your OpenWebUI filters directory

3. Configure your API keys in the OpenWebUI admin panel.

## Configuration

### Setting API Keys

1. Navigate to the OpenWebUI admin panel
2. Go to the "Settings" section
3. Add your API keys:
   - For Anthropic: Add `ANTHROPIC_API_KEY` with your key
   - For DeepSeek: Add `DEEPSEEK_API_KEY` with your key

### Cost Tracker Settings

The Cost Tracker works without additional configuration, but you can customize:

- `compensation`: Set a multiplier for cost calculation (e.g., 1.2 for 20% markup)
- `show_elapsed_time`: Toggle display of processing time
- `show_tokens`: Toggle display of token count
- `show_tokens_per_second`: Toggle display of processing speed
- `debug`: Enable verbose logging for troubleshooting

## Usage Examples

### Using Claude Models

1. Open the chat interface in OpenWebUI
2. Select any Claude model from the model dropdown
3. Start chatting with Claude!

For image understanding:
1. Select a Claude model that supports vision (Claude 3 models)
2. Upload an image
3. Ask questions about the image

### Using DeepSeek Models

1. Open the chat interface in OpenWebUI
2. Select a DeepSeek model from the dropdown
3. Start chatting with DeepSeek!

DeepSeek Reasoner excels at:
- Mathematical problems
- Logical reasoning
- Code generation and analysis

### Monitoring Costs

The Cost Tracker runs automatically in the background:
- Each response will include statistics about time, tokens, and cost
- Costs are logged for future reference and analysis

## Troubleshooting

### Common Issues

#### API Connection Problems
- Ensure your API keys are correctly entered
- Check your internet connection and firewall settings
- Verify API endpoint access (`api.anthropic.com`, `api.deepseek.com`)

#### Cost Tracking Inaccuracies
- Ensure the cost tracker can access the pricing data
- Check for model name mismatches between your system and the pricing JSON
- Enable debug mode for detailed logs

#### Models Not Appearing
- Restart OpenWebUI after installation
- Check the OpenWebUI logs for any errors
- Verify pipe/filter file permissions

### Getting Help

If you encounter issues:
1. Check the OpenWebUI logs for error messages
2. Open an issue on this GitHub repository with detailed information
3. Include relevant log snippets and error messages

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenWebUI](https://openwebui.com) team for the amazing platform
- [Anthropic](https://anthropic.com) for Claude models
- [DeepSeek](https://deepseek.com) for their language models