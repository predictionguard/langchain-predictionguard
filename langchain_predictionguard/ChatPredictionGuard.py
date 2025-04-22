import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, model_validator, Field

from .utils import (
    convert_dict_to_message,
    convert_message_to_dict,
)

logger = logging.getLogger(__name__)


class ChatPredictionGuard(BaseChatModel):
    """Prediction Guard chat models.

    To use, you should have the ``predictionguard`` python package installed,
    and the environment variable ``PREDICTIONGUARD_API_KEY`` set with your API key,
    or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            chat = ChatPredictionGuard(
                predictionguard_api_key="<your API key>",
                model="Hermes-3-Llama-3.1-8B",
            )
    """

    client: Any = None

    model: Optional[str] = "Hermes-3-Llama-3.1-8B"
    """Model name to use."""
    max_tokens: Optional[int] = 256
    """The maximum number of tokens in the generated completion."""
    logit_bias: Optional[dict[int, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion."""
    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens."""
    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency."""
    temperature: Optional[float] = 0.75
    """The temperature parameter for controlling randomness in completions."""
    top_p: Optional[float] = 0.1
    """The diversity of the generated text based on nucleus sampling."""
    top_k: Optional[int] = None
    """The diversity of the generated text based on top-k sampling."""
    stop: Optional[Union[list[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    predictionguard_input: Optional[Dict[str, Union[str, bool]]] = None
    """The input check to run over the prompt before sending to the LLM."""
    predictionguard_output: Optional[Dict[str, bool]] = None
    """The output check to run the LLM output against."""
    predictionguard_api_key: Optional[str] = None
    """Prediction Guard API key."""

    model_config = ConfigDict(extra="forbid")

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``: automatically selects a tool (including no tool).
                - ``"none"``: does not call a tool.
                - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                - ``False`` or ``None``: no effect, default OpenAI behavior.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. If True, the input schema will be
                validated according to
                https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
                If False, input schema will not be validated and model output will not
                be validated.
                If None, ``strict`` argument will not be passed to the model.
            parallel_tool_calls: Set to ``False`` to disable parallel tool use.
                Defaults to ``None`` (no specification, which allows parallel tool use).
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_openai.chat_models.base.ChatOpenAI.bind`.
        """  # noqa: E501

        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in (
                    "file_search",
                    "web_search_preview",
                    "computer_use_preview",
                ):
                    tool_choice = {"type": tool_choice}
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    pass
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "predictionguard-chat"

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        pg_api_key = get_from_dict_or_env(
            values, "predictionguard_api_key", "PREDICTIONGUARD_API_KEY"
        )

        try:
            from predictionguard import PredictionGuard

            values["client"] = PredictionGuard(
                api_key=pg_api_key,
            )

        except ImportError:
            raise ImportError(
                "Could not import predictionguard python package. "
                "Please install it with `pip install predictionguard --upgrade`."
            )

        return values

    def _get_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        # input kwarg conflicts with LanguageModelInput on BaseChatModel
        input = kwargs.pop("predictionguard_input", self.predictionguard_input)
        output = kwargs.pop("predictionguard_output", self.predictionguard_output)

        params = {
            **{
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "logit_bias": self.logit_bias,
                "stop": self.stop,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "input": (
                    input.model_dump() if isinstance(input, BaseModel) else input
                ),
                "output": (
                    output.model_dump() if isinstance(output, BaseModel) else output
                ),
            },
            **kwargs,
        }

        return params

    def _stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = [convert_message_to_dict(m) for m in messages]

        params = self._get_parameters(**kwargs)
        params["stream"] = True

        result = self.client.chat.completions.create(
            model=self.model,
            messages=message_dicts,
            **params,
        )
        for part in result:
            # get the data from SSE
            if "data" in part:
                part = part["data"]
            if len(part["choices"]) == 0:
                continue
            content = part["choices"][0]["delta"]["content"]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(id=part["id"], content=content)
            )
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        params = self._get_parameters(**kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_dicts,
            **params,
        )

        generations = []
        for res in response["choices"]:
            if res.get("status", "").startswith("error: "):
                err_msg = res["status"].removeprefix("error: ")
                raise ValueError(f"Error from PredictionGuard API: {err_msg}")

            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)

        return ChatResult(generations=generations)