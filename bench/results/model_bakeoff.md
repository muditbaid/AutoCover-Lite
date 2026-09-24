# Generator model bake-off: `ticket_price.py`

One Generator round per model, identical cached scenarios. Run 2026-09-24 00:59.

| Model | Candidates | Pass rate | Accepted | Lines | Branches | Latency/call | Tokens | Notes |
|---|---|---|---|---|---|---|---|---|
| `gemini/gemini-3.5-flash` | 12 | 100.0% | 6 | 100.0% | 100.0% | 7.4s | 5709 |  |
| `gemini/gemini-2.5-flash` | 12 | 100.0% | 6 | 100.0% | 100.0% | 16.5s | 9670 |  |
| `ollama_chat/gpt-oss:120b` | 12 | 100.0% | 6 | 100.0% | 100.0% | 9.1s | 3416 |  |
| `cloudflare/@cf/nvidia/nemotron-3-120b-a12b` | 12 | 75.0% | 5 | 100.0% | 100.0% | 11.1s | 4489 |  |
| `ollama_chat/nemotron-3-super` | 12 | 66.7% | 5 | 100.0% | 100.0% | 15.3s | 3856 |  |
| `nvidia_nim/nvidia/nemotron-3-super-120b-a12b` | 12 | 58.3% | 5 | 92.3% | 87.5% | 96.0s | 5585 |  |
| `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | 12 | 58.3% | 5 | 92.3% | 87.5% | 42.2s | 7018 |  |
| `mistral/codestral-2508` | 17 | 52.9% | 5 | 92.3% | 87.5% | 4.7s | 2209 |  |
| `zai/glm-4.7-flash` | 12 | 58.3% | 5 | 38.5% | 12.5% | 38.4s | 7182 |  |
