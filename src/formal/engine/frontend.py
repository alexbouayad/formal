import os
from logging import DEBUG, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import Tensor, nn
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer

from formal.engine.scanner import EMPTY_SCANNER
from formal.grammar.compile import compile_grammar
from formal.language import LanguageInfo

from .buffer import Buffer
from .lexer import Lexer
from .parser import Parser
from .scanner import ExternalScanner

if TYPE_CHECKING:
    from transformers import TokenizersBackend

logger = getLogger(__name__)


class FormalEngine:
    __slots__ = ("_buffer", "_parser", "_tokenizer")

    _buffer: Final[Buffer]
    _parser: Final[Parser]
    _tokenizer: Final["TokenizersBackend"]

    def __init__(self, language_info: LanguageInfo, tokenizer: "TokenizersBackend") -> None:
        buffer = Buffer()

        text_tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokenizer.get_vocab()]
        grammar = compile_grammar(language_info.ts_grammar_path, text_tokens)

        lexer = Lexer(
            lex_tables=grammar.lex_tables,
            end_symbol=grammar.end_symbol,
            end_of_nonterminal_extra_symbol=grammar.end_of_nonterminal_extra_symbol,
            buffer=buffer,
        )

        scanner = (
            ExternalScanner(
                language_info=language_info,
                external_symbols=grammar.external_symbols,
                buffer=buffer,
            )
            if Path(language_info.ts_scanner_path).exists()
            else EMPTY_SCANNER
        )

        parser = Parser(
            parse_tables=grammar.parse_tables,
            buffer=buffer,
            lexer=lexer,
            scanner=scanner,
        )

        self._buffer = buffer
        self._parser = parser
        self._tokenizer = tokenizer

    # TODO
    def evaluate(
        self,
        source: str,
        # model: TransformerModel,
        # input_ids: Tensor,
        *,
        # num_samples: int = 10,
        # temperature: float = 1.0,
        is_eos: bool = False,
    ) -> bool:
        token_ids = self._tokenizer.encode(source)  # type: ignore
        sequence_length = len(token_ids)

        # sequence_length = input_ids.shape[1]
        # vocab_size = len(self.vocab)

        # mask_shape = (sequence_length - 1, vocab_size)
        # invalid_mask = torch.ones(mask_shape, dtype=torch.bool, device=model.device)

        # outputs = model(input_ids)
        # logits = outputs.logits
        # assert logits is not None

        # scores = logits[0, ...] / temperature
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        # sampled_ids = torch.multinomial(probs, num_samples, replacement=True)

        # log_valid_prob = 0.0

        token_id = token_ids[0]

        # input_id = input_ids[0, 0].item()
        # assert isinstance(input_id, int)
        is_valid = self._evaluate(token_id)

        for sequence_index in range(1, sequence_length):
            token_id = token_ids[sequence_index]

            # input_id = input_ids[0, sequence_index].item()
            # assert isinstance(input_id, int)

            # backup_queue = self._paused_heads.clone()
            # valid_prob = probs[sequence_index - 1, input_id].item() / (num_samples + 1)

            # for sample_index in range(num_samples):
            #     sampled_id = sampled_ids[sequence_index - 1, sample_index].item()
            #     assert isinstance(sampled_id, int)

            #     self._paused_queue = backup_queue.clone()
            #     self._evaluate(sampled_id)

            #     if self._paused_queue:
            #         if self.debug:
            #             logger.debug("SAMPLED TOKEN VALIDATED)")

            #         valid_prob += 1 / (num_samples + 1)

            #     else:
            #         if self.debug:
            #             logger.debug("SAMPLED TOKEN REJECTED)")

            # log_valid_prob += math.log(valid_prob)

            # sampled_tokens: list[str] = []

            # for sample_index in range(3):
            #     sampled_id = sampled_ids[sequence_index - 1, sample_index].item()
            #     assert isinstance(sampled_id, int)
            #     sampled_tokens.append(repr(self.vocab[sampled_id][:10]))

            # print(
            #     sequence_index,
            #     valid_prob,
            #     math.exp(log_valid_prob),
            #     repr(self.vocab[input_id]),
            #     sampled_tokens,
            # )

            # self._paused_heads = backup_queue.clone()

            is_valid = self._evaluate(token_id)

        if is_eos:
            is_valid = self._evaluate(None)

        # input_tensor = tensor([input_ids], device=model.device)
        # outputs = model(input_tensor[:, :-1])
        # targets = input_tensor[0, 1:]

        # logits = outputs.logits
        # assert logits is not None
        # logits = logits[0, :]

        # log_loss = torch.nn.functional.cross_entropy(logits, targets).item()
        # return log_loss

        return is_valid

    def generate(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        prompt: str,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        custom_streamer: BaseStreamer | None = None,
        **model_kwargs: Any,
    ) -> Tensor:
        buffer = self._buffer
        parser = self._parser

        buffer.feed(prompt)
        parser.parse()

        # init values
        pad_token_id = generation_config._pad_token_tensor  # type: ignore
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)  # type: ignore

        model_forward = model.forward
        compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)  # type: ignore
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if model.config._attn_implementation == "flash_attention_2":  # type: ignore
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    print(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`.",
                        end="\n",
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = model.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:  # type: ignore
            model_kwargs = model._prefill_chunking(input_ids, generation_config, **model_kwargs)  # type: ignore
            is_prefill = False
        else:
            is_prefill = True

        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):  # type: ignore
            # prepare model inputs
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)  # type: ignore

            if is_prefill:
                outputs = model(**model_inputs, return_dict=True)  # type: ignore
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)  # type: ignore

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = model._update_model_kwargs_for_generation(  # type: ignore
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)  # type: ignore

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)  # type: ignore

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)  # type: ignore
                if output_logits:
                    raw_logits += (next_token_logits,)  # type: ignore
                if output_attentions:
                    decoder_attentions += (  # type: ignore
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)  # type: ignore
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)  # type: ignore

                if output_hidden_states:
                    decoder_hidden_states += (  # type: ignore
                        (outputs.decoder_hidden_states,)  # type: ignore
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)  # type: ignore
                    )

            parser.checkpoint()

            while True:
                if next_token_scores.max() == -torch.inf:
                    next_token_scores = logits_processor(input_ids, next_token_logits)

                    if logger.isEnabledFor(DEBUG):
                        logger.debug("ALL VOCAB TOKENS REJECTED, REPROCESSING LOGITS")
                        logger.debug("---------------------------------------------")
                        logger.debug("")
                        logger.debug("")
                        logger.debug("")

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                next_input_id = next_tokens.item()
                next_input_id = cast(int, next_input_id)

                self._evaluate(next_input_id)

                if parser.is_live:
                    buffer.validate()

                    if logger.isEnabledFor(DEBUG):
                        logger.debug("VOCAB TOKEN VALIDATED")
                        logger.debug("---------------------")
                        logger.debug("")
                        logger.debug("")
                        logger.debug("")

                    break

                next_token_logits[0, next_input_id] = -torch.inf
                next_token_scores[0, next_input_id] = -torch.inf

                buffer.revert()
                parser.revert()

                if logger.isEnabledFor(DEBUG):
                    logger.debug("VOCAB TOKEN REJECTED")
                    logger.debug("--------------------")
                    logger.debug("")
                    logger.debug("")
                    logger.debug("")

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)  # type: ignore

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  # type: ignore
            if custom_streamer is not None:
                custom_streamer.put(next_tokens.cpu())  # type: ignore

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if custom_streamer is not None:
            custom_streamer.end()

        return input_ids

    def parse(self, text: str, *, is_eos: bool = True) -> bool:
        self._buffer.feed(text, is_eos=is_eos)

        return self._parser.parse(is_eos=is_eos)

    def reset(self) -> None:
        self._buffer.clear()
        self._parser.reset()

    def _evaluate(self, token_id: int | None) -> bool:
        if token_id is None:
            text = ""
            is_eos = True

        else:
            text = cast(str, self._tokenizer.decode(token_id, clean_up_tokenization_spaces=False))  # type: ignore
            is_eos = False

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"TEXT: {text!r} (input_id={token_id})")

        self._buffer.feed(text, is_eos=is_eos)

        return self._parser.parse(is_eos=is_eos)
