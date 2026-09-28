# langchain-predictionguard

> [!WARNING]
> **This package is deprecated and no longer maintained.** Some features are broken or missing, and no further updates will be released. Existing releases remain installable from PyPI, but you should migrate to `langchain-openai` or `langchain-anthropic` as described below.

## Migrating

The Prediction Guard API is compatible with both OpenAI-style and Anthropic-style clients, so the standard LangChain integrations work with it directly. Use whichever matches the functionality you need, pointed at the Prediction Guard API with your existing API key.

| This package | Replacement |
|---|---|
| `ChatPredictionGuard` | `langchain_openai.ChatOpenAI` or `langchain_anthropic.ChatAnthropic` |
| `PredictionGuard` (completions) | `langchain_openai.OpenAI` |
| `PredictionGuardEmbeddings` | `langchain_openai.OpenAIEmbeddings` |
| `PredictionGuardRerank` | No drop-in replacement; call the Prediction Guard `/rerank` endpoint directly |

### Chat (OpenAI-compatible)

```bash
pip install langchain-openai
```

```python
import os

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="<model-name>",
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
)

chat.invoke("Tell me a joke")
```

### Chat (Anthropic-compatible)

```bash
pip install langchain-anthropic
```

```python
import os

from langchain_anthropic import ChatAnthropic

chat = ChatAnthropic(
    model="<model-name>",
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
)

chat.invoke("Tell me a joke")
```

### Completions

```python
import os

from langchain_openai import OpenAI

llm = OpenAI(
    model="<model-name>",
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
)

llm.invoke("Tell me a joke about bears")
```

### Embeddings

```python
import os

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="<model-name>",
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
    # Send raw text rather than OpenAI tiktoken token IDs.
    check_embedding_ctx_length=False,
)

embeddings.embed_query("This is an embedding example.")
```

For the full list of endpoints and models, see the [Prediction Guard documentation](https://docs.predictionguard.com).
