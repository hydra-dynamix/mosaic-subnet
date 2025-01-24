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

The Mosaic miner supports multiple providers for image generation:

### Available Providers

1. **Local Provider** (Default)
   - Uses local GPU/CPU for inference
   - Supports models from HuggingFace (default: stabilityai/sdxl-turbo)
   - No API key required
   ```bash
   python -m mosaic_subnet.cli miner your-key --provider local --model stabilityai/sdxl-turbo
   ```

2. **OpenAI Provider**
   - Uses DALL-E 2 or DALL-E 3
   - Requires OpenAI API key
   ```bash
   python -m mosaic_subnet.cli miner your-key --provider openai --api-key your-api-key --model dall-e-3
   ```

3. **Custom Provider**
   - Use any custom inference endpoint
   - Supports configurable headers and parameters
   ```bash
   python -m mosaic_subnet.cli miner your-key --provider custom --api-url your-api-url --api-key your-api-key
   ```

### Configuration Options

All settings can be configured via environment variables or a config file:

1. Copy the sample configuration:
```bash
cp env/sample.config.env env/config.env
```

2. Edit the configuration:
```bash
nano env/config.env
```

#### Provider Settings
- `MOSAIC_MINER_PROVIDER`: Provider type (local, openai, custom)
- `MOSAIC_MINER_MODEL`: Model identifier
- `MOSAIC_MINER_API_KEY`: API key for OpenAI/custom providers
- `MOSAIC_MINER_API_BASE_URL`: Base URL for custom provider

#### Network Settings
- `MOSAIC_MINER_HOST`: Host to bind to (default: 0.0.0.0)
- `MOSAIC_MINER_PORT`: Port to run on (default: 8080)
- `MOSAIC_USE_TESTNET`: Use testnet endpoints (default: false)
- `MOSAIC_CALL_TIMEOUT`: API call timeout in seconds (default: 60)

#### OpenAI-specific Settings
- `MOSAIC_OPENAI_SIZE`: Image size (1024x1024, 1024x1792, or 1792x1024)
- `MOSAIC_OPENAI_QUALITY`: Image quality for DALL-E 3 (standard or hd)
- `MOSAIC_OPENAI_STYLE`: Image style for DALL-E 3 (vivid or natural)

#### Custom Provider Settings
- `MOSAIC_MINER_API_ADDITIONAL_HEADERS`: Additional HTTP headers as JSON
- `MOSAIC_MINER_API_ADDITIONAL_PARAMS`: Additional API parameters as JSON

### Provider Capabilities

Each provider has different capabilities:

1. **Local Provider**
   - ✅ Negative prompts
   - ✅ Seed control
   - ✅ Step count control
   - ❌ Size/quality/style options

2. **OpenAI Provider**
   - ✅ Negative prompts (via prompt modification)
   - ❌ Seed control
   - ❌ Step count control
   - ✅ Size/quality/style options (DALL-E 3)

3. **Custom Provider**
   - Capabilities depend on your API
   - Configurable via additional parameters

## Links
- Website: [mosaicx.org](https://mosaicx.org/)
- Docs: [docs.mosaicx.org](https://docs.mosaicx.org/)
- Leader Board: [leaderboard.mosaicx.org](https://leaderboard.mosaicx.org/)
- GitHub: [mosaic-subnet](https://github.com/mosaicx-org/mosaic-subnet)
