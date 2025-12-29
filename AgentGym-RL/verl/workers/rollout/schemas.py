from dataclasses import dataclass, field
from typing import Any, Literal
from transformers import PreTrainedTokenizer # type: ignore
import torch


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


@dataclass
class Message:
    role: str
    content: str

    # Strict per-message attributes for logging/analysis (always serialized).
    step_idx: int | None = None
    kind: str | None = None
    phase: str | None = None

    raw_model_output: str | None = None
    parsed_action: str | None = None
    action: str | None = None

    reward: float | None = None
    done: bool | None = None

    step_score: float | None = None
    step_score_delta: float | None = None

    is_valid_action: bool | None = None
    valid_actions: list[str] = field(default_factory=list)

    # Retrieval-augmented memory (strict fields; present on every message).
    memory_used: bool = False
    memory_query: str | None = None
    memory_retrieved: list[dict[str, Any]] = field(default_factory=list)
    memory_encoder: str | None = None
    memory_task_specific: bool | None = None
    memory_k: int | None = None


    def to_chat_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "step_idx": self.step_idx,
            "kind": self.kind,
            "phase": self.phase,
            "raw_model_output": self.raw_model_output,
            "parsed_action": self.parsed_action,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "step_score": self.step_score,
            "step_score_delta": self.step_score_delta,
            "is_valid_action": self.is_valid_action,
            "valid_actions": self.valid_actions,
            "memory_used": self.memory_used,
            "memory_query": self.memory_query,
            "memory_retrieved": self.memory_retrieved,
            "memory_encoder": self.memory_encoder,
            "memory_task_specific": self.memory_task_specific,
            "memory_k": self.memory_k,
        }

    def to_dict(self) -> dict[str, str]:
        '''Backwards-compatible alias for chat-template dicts.'''
        return self.to_chat_dict()

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __str__(self) -> str:
        return self.__repr__()


QWEN_FORMAT_CONFIG: dict[str, str] = {
    'assistant_prefix': '\n<|im_start|>assistant\n',
    'assistant_suffix': '<|im_end|>',
    'user_prefix': '\n<|im_start|>user\n',
    'user_suffix': '<|im_end|>',
}


@dataclass
class RolloutEpisode:
    '''Tracks the state of a single agent episode during rollout.'''

    # Episode identity
    task_name: str
    item_id: int

    # Conversation state
    messages: list[Message]

    # Episode status
    score: float = 0.0
    done: bool = False

    # Token sequences (grow during rollout)
    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    # Initial prompt snapshots (for extracting response later)
    prompt_ids: list[int] = field(default_factory=list)
    prompt_attention_mask: list[int] = field(default_factory=list)
    prompt_position_ids: list[int] = field(default_factory=list)
    prompt_loss_mask: list[int] = field(default_factory=list)

    # Response sequences (populated by truncate_output_ids)
    response_ids: list[int] = field(default_factory=list)
    response_attention_mask: list[int] = field(default_factory=list)
    response_position_ids: list[int] = field(default_factory=list)
    response_loss_mask: list[int] = field(default_factory=list)

    # Step-level metrics
    step_rewards: list[float] = field(default_factory=list)
    step_valid_actions: list[bool] = field(default_factory=list)
    step_scores: list[float] = field(default_factory=list)
    step_valid_action_strings: list[list[str]] = field(default_factory=list)
    step_parsed_actions: list[str] = field(default_factory=list)
    step_agent_actions: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    # Limits
    max_response_len: int = 8192
    max_model_len: int = 32768

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer, memory_examples: list[dict] | None = None) -> list[int]:
        '''
        Generate prompt with optional memory examples injected before current conversation.

        Args:
            tokenizer: Tokenizer for encoding
            memory_examples: Optional list of few-shot examples from memory bank
                            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            Tokenized prompt with memory examples prepended
        '''
        conversations = [msg.to_chat_dict() for msg in self.messages]

        # Inject memory examples before current conversation if provided
        if memory_examples and len(memory_examples) > 0:
            # Find system message if it exists
            system_msg = None
            current_conv = conversations
            if conversations and conversations[0].get("role") == "system":
                system_msg = conversations[0]
                current_conv = conversations[1:]

            # Build conversation with memory examples
            if system_msg:
                conversations = [system_msg] + memory_examples + current_conv
            else:
                conversations = memory_examples + current_conv

        return tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True) # type: ignore


    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal['qwen'] = 'qwen',
        step_idx: int | None = None,
        kind: str | None = None,
        phase: str | None = None,
        raw_model_output: str | None = None,
        parsed_action: str | None = None,
        action: str | None = None,
        valid_actions: list[str] | None = None,
        is_valid_action: bool | None = None,
        reward: float | None = None,
        done: bool | None = None,
        step_score: float | None = None,
        step_score_delta: float | None = None,
        memory_used: bool = False,
        memory_query: str | None = None,
        memory_retrieved: list[dict[str, Any]] | None = None,
        memory_encoder: str | None = None,
        memory_task_specific: bool | None = None,
        memory_k: int | None = None,
    ) -> None:
        msg = Message(
            role='assistant',
            content=content,
            step_idx=step_idx,
            kind=kind,
            phase=phase,
            raw_model_output=raw_model_output,
            parsed_action=parsed_action,
            action=action,
            reward=reward,
            done=done,
            step_score=step_score,
            step_score_delta=step_score_delta,
            is_valid_action=is_valid_action,
            valid_actions=[] if valid_actions is None else valid_actions,
            memory_used=memory_used,
            memory_query=memory_query,
            memory_retrieved=[] if memory_retrieved is None else memory_retrieved,
            memory_encoder=memory_encoder,
            memory_task_specific=memory_task_specific,
            memory_k=memory_k,
        )
        self.messages.append(msg)
        assert format == 'qwen', f'format {format} not supported'
        prefix_msg = QWEN_FORMAT_CONFIG['assistant_prefix']
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = QWEN_FORMAT_CONFIG['assistant_suffix']
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(f'Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}')
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f'Rollout Handler has different length of {len(self.input_ids)=}, {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}'

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal['qwen'] = 'qwen',
        step_idx: int | None = None,
        kind: str | None = None,
        phase: str | None = None,
        raw_model_output: str | None = None,
        parsed_action: str | None = None,
        action: str | None = None,
        valid_actions: list[str] | None = None,
        is_valid_action: bool | None = None,
        reward: float | None = None,
        done: bool | None = None,
        step_score: float | None = None,
        step_score_delta: float | None = None,
        memory_used: bool = False,
        memory_query: str | None = None,
        memory_retrieved: list[dict[str, Any]] | None = None,
        memory_encoder: str | None = None,
        memory_task_specific: bool | None = None,
        memory_k: int | None = None,
    ) -> None:
        msg = Message(
            role='user',
            content=content,
            step_idx=step_idx,
            kind=kind,
            phase=phase,
            raw_model_output=raw_model_output,
            parsed_action=parsed_action,
            action=action,
            valid_actions=[] if valid_actions is None else valid_actions,
            is_valid_action=is_valid_action,
            reward=reward,
            done=done,
            step_score=step_score,
            step_score_delta=step_score_delta,
            memory_used=memory_used,
            memory_query=memory_query,
            memory_retrieved=[] if memory_retrieved is None else memory_retrieved,
            memory_encoder=memory_encoder,
            memory_task_specific=memory_task_specific,
            memory_k=memory_k,
        )
        self.messages.append(msg)
        assert format == 'qwen', f'format {format} not supported'
        prefix_msg = QWEN_FORMAT_CONFIG['user_prefix']
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = QWEN_FORMAT_CONFIG['user_suffix']
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(f'Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}')

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f'Rollout Handler has different length of {len(self.input_ids)=}, {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}'

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
