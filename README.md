![image](https://github.com/mosaicx-org/mosaic-subnet/assets/48199614/97ecdea5-7c35-4536-9014-ac85da575974)


# Mosaic Subnet
Mosaic is an open platform for generative artificial intelligence. Users can generate images from natural language descriptions using modules provided by [Commune AI](https://communeai.org/).


## Design
Within the Commune subnet, Mosaic endeavors to foster greater engagement among model developers in the image AIGC network through equitable incentivization mechanisms. Additionally, the aim is to enhance accessibility to these exemplary AI models for a broader user base via API integration and a dedicated web application.

<img width="981" alt="image" src="https://github.com/mosaicx-org/mosaic-subnet/assets/48199614/0fde2ff5-0eee-46a3-b615-942c4717723c">

Within the Mosaic subnet, two distinct task categories exist:
- The first entails requests originating from the application layer, routed through an HTTP gateway for subsequent processing by validators.
- The second involves prompts sourced from datasets, intended for ongoing task dissemination to miners for computation and validation during periods of system idleness.

The validator will extract the embedding vectors from the text prompt and the generated image, then employ cosine similarity to assess the likeness between the two embedding vectors. This allows us to ascertain whether the generated image aligns with the descriptive content of the prompt text.


## Guide
For guidance for miners and validators, please refer to. [Quick Start Guide](docs/quickstart.md)

## Miner Configuration

The Mosaic miner supports two provider types for image generation:

### Available Providers

1. **Local Provider**
   - Uses local GPU/CPU for inference with diffusers
   - Supports models from HuggingFace (default: stabilityai/sdxl-turbo)
   - No API key required
   ```bash
   python -m mosaic_subnet.cli miner your-key --provider local
   ```

2. **OpenAI Provider**
   - For custom API endpoints that follow OpenAI's format
   - Requires API URL and key
   ```bash
   python -m mosaic_subnet.cli miner your-key --provider openai --api-url your-api-url --api-key your-api-key
   ```

### Configuration Options

Settings can be configured via environment variables or a config file:

1. Copy the sample configuration:
```bash
cp env/sample.config.env env/config.env
```

2. Edit the configuration:
```bash
nano env/config.env
```

#### Provider Settings
- `MOSAIC_MINER_PROVIDER`: Provider type (`local` or `openai`)
- `MOSAIC_MINER_MODEL`: Model identifier (for local provider)
- `MOSAIC_MINER_API_URL`: API URL (for openai provider)
- `MOSAIC_MINER_API_KEY`: API key (for openai provider)

#### Network Settings
Network settings like host and port have sensible defaults and can be overridden via CLI arguments if needed:
```bash
python -m mosaic_subnet.cli miner your-key --host 127.0.0.1 --port 8000
```

### Provider Capabilities

Each provider has different capabilities:

1. **Local Provider**
   - Uses HuggingFace's diffusers library
   - Supports various Stable Diffusion models
   - Default model: stabilityai/sdxl-turbo
   - Requires GPU for reasonable performance

2. **OpenAI Provider**
   - For custom endpoints that follow OpenAI's API format
   - Endpoint must support:
     - POST /v1/images/generations
     - Request format: `{"prompt": "...", "n": 1, "response_format": "url"}`
     - Response format: `{"data": [{"url": "..."}]}`

## Links
- Website: [mosaicx.org](https://mosaicx.org/)
- Docs: [docs.mosaicx.org](https://docs.mosaicx.org/)
- Leader Board: [leaderboard.mosaicx.org](https://leaderboard.mosaicx.org/)
- GitHub: [mosaic-subnet](https://github.com/mosaicx-org/mosaic-subnet)
