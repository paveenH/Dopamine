import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig
from diffusion import diffusion_generate

log = logging.getLogger(__name__)


def _is_mistral3_model(model_path: str) -> bool:
    """Check if the model is a Mistral3 multimodal model."""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config.model_type == "mistral3"
    except Exception:
        # Fallback: check model path name
        return "mistral3" in model_path.lower() or "ministral-3" in model_path.lower()


class VicundaModel:
    """
    Wrapper around a CausalLM to provide a consistent interface,
    support for quantization, multi–GPU loading, and role–based prompts.
    """

    task: str = "text2text-generation"

    def __init__(
        self,
        model_path: str,
        diffusion_mode: str = None,  # whether to use diffusion with dream mode
    ) -> None:
        self.model_path = model_path
        self.diffusion_mode = diffusion_mode

        # Counts prefill-hook injections that ACTUALLY fired (see
        # _regenerate_prefill_only). Purely observational: callers that want to
        # attest steering read it via steering_fire_count(); no behaviour on any
        # path depends on its value.
        self._steering_fire_count = 0

        # Model
        if diffusion_mode == "dream":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # or use torch.float32 if needed
                device_map="auto",
            )
        elif _is_mistral3_model(model_path):
            # Mistral3 is a multimodal model, requires specific class
            from transformers import Mistral3ForConditionalGeneration
            log.info(f"Detected Mistral3 model, using Mistral3ForConditionalGeneration")
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # or use torch.float32 if needed
                device_map="auto",
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        # Llama3 is BPE; clean_up_tokenization_spaces=True is destructive for BPE
        # (strips spaces before punctuation) and newer transformers defaults it
        # True + warns. Force False so decode preserves spacing — this matches
        # the (silent) behavior under the older transformers the GSM8K results
        # were produced with, keeping GSM8K/MATH numbers comparable, and avoids
        # corrupting LaTeX/spacing in MATH generations. Set as an attribute so it
        # covers every self.tokenizer.decode() call site.
        self.tokenizer.clean_up_tokenization_spaces = False
        self._ensure_padding_token()

    # ───────────────────── Core helpers ───────────────────── #

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.terminators = self._build_terminators()

    def _build_terminators(self) -> list[int]:
        """Stop-token id list for generation.

        Llama-3.x-Instruct ends each assistant turn with <|eot_id|> (128009),
        NOT <|end_of_text|> (= tokenizer.eos_token, 128001). Passing only
        eos_token_id means the model "wants to stop" (emits <|eot_id|>) but that
        token is not a terminator, so decoding runs to max_new_tokens and the
        tail degenerates into a repetition loop. Including <|eot_id|> lets the
        chat-formatted model stop naturally. Harmless for non-chat models: the
        token either is absent (skipped) or never generated.

        Qwen (2.5 / 3) ends its turn with <|im_end|>, and ships a
        generation_config whose eos_token_id is a LIST (e.g. [151645, 151643]);
        tokenizer.eos_token_id alone can miss one of them, which reproduces the
        same run-to-max_new_tokens loop. So we also union in the model's own
        generation_config eos ids.
        """
        ids = [self.tokenizer.eos_token_id]
        for tok in ("<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            unk = getattr(self.tokenizer, "unk_token_id", None)
            if tid is not None and tid != unk and tid not in ids:
                ids.append(tid)

        # Union in generation_config.eos_token_id (int or list); Qwen ships a list.
        gen_cfg = getattr(self.model, "generation_config", None)
        cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if cfg_eos is not None:
            for tid in (cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]):
                if isinstance(tid, int) and tid not in ids:
                    ids.append(tid)

        return [t for t in ids if t is not None]

    def _find_decoder_layers(self):
        """
        Collect all decoder layers. Supports multiple naming conventions:
        - Standard CausalLM: 'model.layers.*'
        - Mistral3 (multimodal): 'model.language_model.layers.*'
        Called by all hook functions to preserve original behavior.
        """
        # Try standard CausalLM naming first
        decoder_layers = [
            module for name, module in self.model.named_modules()
            if name.startswith("model.layers.") and name.count(".") == 2
        ]

        # Try Mistral3/multimodal naming (model.language_model.layers.*)
        if not decoder_layers:
            decoder_layers = [
                module for name, module in self.model.named_modules()
                if name.startswith("model.language_model.layers.") and name.count(".") == 3
            ]

        if not decoder_layers:
            # Print all module names for debugging
            print("Available module names:")
            for name, module in self.model.named_modules():
                if "layer" in name.lower():
                    print(f"  {name}")
            raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")
        return decoder_layers

    # ───────────────────── Hook framework ───────────────────── #

    def _apply_diff_hooks(
        self,
        diff_matrices: list[np.ndarray],
        forward_fn,
        last_indices: torch.Tensor | None = None,
        tail_len: int = 1,
    ):
        """
        Add diff_matrices to last token (or tail_len tokens) for each layer.
        """
        decoder_layers = self._find_decoder_layers()

        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"Number of difference matrices ({len(diff_matrices)}) "
                f"does not match number of decoder layers ({len(decoder_layers)})."
            )

        def create_hook(diff_matrix):
            # Computed ONCE per layer, outside the hook: diff_matrices is
            # full-length (all 32 decoder layers) with zero rows outside the
            # steered band, so an all-zero layer adds nothing and must not be
            # counted as an injection. Checking it per forward would be a
            # per-step host sync on a tensor that never changes.
            _layer_is_steered = bool(np.any(np.asarray(diff_matrix) != 0))

            def hook(module, input, output):
                def prepare_diff(hs: torch.Tensor) -> torch.Tensor:
                    B, _, H = hs.shape
                    diff_t = torch.as_tensor(diff_matrix, device=hs.device, dtype=hs.dtype)
                    if diff_t.ndim == 1:
                        diff_t = diff_t.unsqueeze(0).expand(B, -1)  # expand to [B, H]
                    elif diff_t.ndim == 2 and diff_t.shape[0] == 1:
                        diff_t = diff_t.expand(B, -1)  # expand [1, H] to [B, H]
                    else:
                        assert diff_t.shape == (B, H), f"diff shape {diff_t.shape} != (B, {H})"
                    return diff_t  # return [B, H]

                def add_at_tail(hs: torch.Tensor) -> torch.Tensor:
                    # hs: [B, L, H] (batch, sequence_length, hidden_size)
                    B, L, H = hs.shape
                    n = max(int(tail_len), 1)

                    if last_indices is not None:
                        last_pos = last_indices.to(device=hs.device, dtype=torch.long)  # [B]
                    else:
                        last_pos = torch.full((B,), L - 1, device=hs.device, dtype=torch.long)  # [B]

                    diff_bh = prepare_diff(hs)  # [B, H]

                    # In-place add at target positions (avoid allocating [B, L, H] buffer)
                    for t in range(n):
                        pos = last_pos - t  # [B]
                        valid = pos >= 0    # [B]
                        if not valid.any():
                            break
                        pos_clamped = pos.clamp_min(0)  # [B]
                        batch_idx = torch.arange(B, device=hs.device)
                        valid_b = batch_idx[valid]
                        valid_p = pos_clamped[valid]
                        hs[valid_b, valid_p, :] += diff_bh[valid]
                        # Observed-injection counter; see steering_fire_count().
                        # Counts SITES (sequence x token position), not hook
                        # calls: diff_matrices is full-length with zeros outside
                        # the steered band, and a K-candidate batch is ONE
                        # forward, so counting calls would report the same
                        # number for every layer and every K. Skipping all-zero
                        # layers is what makes the total mean "actual non-zero
                        # injections".
                        if _layer_is_steered:
                            self._steering_fire_count += int(valid.sum())

                    return hs

                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states = add_at_tail(hidden_states)
                    return (hidden_states,) + output[1:]
                else:
                    hidden_states = output
                    hidden_states = add_at_tail(hidden_states)
                    return hidden_states

            return hook

        hooks = []
        for layer, diff_matrix in zip(decoder_layers, diff_matrices):
            hook = layer.register_forward_hook(create_hook(diff_matrix))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

    def _apply_replace_hooks(self, replace_matrices: list[np.ndarray], forward_fn, start: int = 0, end: int = None):
        """
        Replace the last token's hidden state with replace_matrices for layers in [start, end).
        """
        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if end is None or end > num_layers:
            end = num_layers

        if len(replace_matrices) < num_layers:
            raise ValueError(
                f"replace_matrices has length {len(replace_matrices)}, "
                f"but we found {num_layers} decoder layers. Need at least >= num_layers."
            )
        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}. Must be in [0, {num_layers-1}].")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}. Must have end > start.")

        def create_replace_hook(replace_matrix: np.ndarray):
            def hook(module, module_input, module_output):
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]
                    last_token_idx = hidden_states.shape[1] - 1
                    if replace_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Replacement hidden_size ({replace_matrix.shape[-1]}) "
                            f"!= model hidden_size ({hidden_states.shape[-1]})."
                        )
                    rep_tensor = torch.tensor(replace_matrix, device=hidden_states.device).unsqueeze(0)
                    hidden_states[:, last_token_idx, :] = rep_tensor
                    return (hidden_states,) + module_output[1:]
                else:
                    last_token_idx = module_output.shape[1] - 1
                    if replace_matrix.shape[-1] != module_output.shape[-1]:
                        raise ValueError(
                            f"Replacement hidden_size ({replace_matrix.shape[-1]}) "
                            f"!= model hidden_size ({module_output.shape[-1]})."
                        )
                    rep_tensor = torch.tensor(replace_matrix, device=module_output.device).unsqueeze(0)
                    module_output[:, last_token_idx, :] = rep_tensor
                    return module_output

            return hook

        hooks = []
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer = decoder_layers[layer_idx]
                rep_matrix = replace_matrices[layer_idx]
                hook = layer.register_forward_hook(create_replace_hook(rep_matrix))
                hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for hook in hooks:
                hook.remove()

        return outputs

    def _apply_index_lesion_hooks(self, neuron_indices: list[int], forward_fn, start: int = 0, end: int = None):
        """
        Zero out given neuron_indices for layers in [start, end).
        """
        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if end is None or end > num_layers:
            end = num_layers
        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}, must be in [0, {num_layers-1}]")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}, must have end>start")

        def create_lesion_hook(neuron_ids: list[int]):
            def hook(module, module_input, module_output):
                if len(neuron_ids) == 0:
                    return module_output
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]
                    if hidden_states.shape[-1] <= max(neuron_ids):
                        raise ValueError("Some neuron index is out of range for the hidden_size.")
                    hidden_states[..., neuron_ids] = 0.0
                    return (hidden_states,) + module_output[1:]
                else:
                    if module_output.shape[-1] <= max(neuron_ids):
                        raise ValueError("Some neuron index is out of range for the hidden_size.")
                    module_output[..., neuron_ids] = 0.0
                    return module_output

            return hook

        hooks = []
        for layer_idx in range(start, end):
            layer = decoder_layers[layer_idx]
            hook = layer.register_forward_hook(create_lesion_hook(neuron_indices))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def _apply_rsn_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
        mode: str = "lesion",  # "lesion" or "complement"
    ):
        """
        - lesion:
            rsn_ids = []     → do nothing for this layer
            rsn_ids = [ids]  → zero-out ONLY these neurons
            
        - complement:
            rsn_ids = []     → zero-out ENTIRE layer
            rsn_ids = [ids]  → keep-only these neurons, zero-out others
        """

        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if len(rsn_indices_per_layer) != num_layers:
            raise ValueError(
                f"rsn_indices_per_layer has {len(rsn_indices_per_layer)}, "
                f"but model has {num_layers} layers."
            )

        def create_layer_hook(rsn_ids):
            rsn_ids = np.array(rsn_ids, dtype=int)

            def hook(module, module_input, module_output):

                if isinstance(module_output, tuple):
                    hs = module_output[0]
                    tail = module_output[1:]
                else:
                    hs = module_output
                    tail = None

                H = hs.shape[-1]

                # -------- LESION mode --------
                if mode == "lesion":
                    # if rsn_ids is empty, skip; otherwise zero out specified neurons
                    if rsn_ids.size > 0:
                        hs[..., rsn_ids] = 0.0

                # -------- COMPLEMENT mode --------
                elif mode == "complement":

                    # if rsn_ids is empty, zero out entire layer; otherwise keep only rsn_ids
                    if rsn_ids.size == 0:
                        hs[..., :] = 0.0

                    else:
                        # keep only rsn_ids, zero out all others
                        drop_ids = np.setdiff1d(np.arange(H), rsn_ids)
                        if drop_ids.size > 0:
                            hs[..., drop_ids] = 0.0

                else:
                    raise ValueError(f"Unknown mode={mode}")

                if tail is None:
                    return hs
                return (hs,) + tail

            return hook

        # Register hooks for each layer
        hooks = []
        for L in range(num_layers):
            rsn_ids = rsn_indices_per_layer[L]   # per-layer neuron indices list
            hook = decoder_layers[L].register_forward_hook(create_layer_hook(rsn_ids))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def _apply_rsn_lesion_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
    ):
        return self._apply_rsn_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
            mode="lesion",
        )

    def _apply_rsn_complement_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
    ):
        return self._apply_rsn_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
            mode="complement",
        )

    # ───────────────────── Generate Logits ───────────────────── #

    @torch.no_grad()
    def get_logits(
        self, prompts: list[str], return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        tokens = tokens.to(self.model.device)

        output = self.model(**tokens, return_dict=True, output_hidden_states=return_hidden, use_cache=False)

        if return_hidden:
            return output.logits, output.hidden_states
        return output.logits

    @torch.no_grad()
    def regenerate_logits(self, prompts: list[str], diff_matrices: list[np.ndarray], tail_len: int = 1):
        if diff_matrices is None:
            raise ValueError("diff_matrices required")

        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(self.model.device)
        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(**tokens, return_dict=True, output_hidden_states=False, use_cache=False).logits

        full_logits = self._apply_diff_hooks(diff_matrices, forward_fn, last_indices=last_idx, tail_len=tail_len)  # shape: (B, L, V)

        B, L, V = full_logits.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=idx).squeeze(1)  # shape: (B, V)
        return last_logits.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def regenerate_logits_teacher_forcing(
        self,
        prompts: list[str],
        answer_token_ids: list[list[int]],
        diff_matrices: list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        """
        Teacher-forcing joint logit extraction with optional RSN steering.

        For each sample i:
          - Input sequence: [prompt_tokens_i] + [answer_token_ids_i]
          - Steering (if diff_matrices): applied ONLY at the last prompt token position
          - Returns: for each answer token position k, the logit vector at position
                     (prompt_end + k - 1), i.e. the position that predicts token k.

        Returns: list of length B, each element is ndarray of shape (n_answer_tokens, V).
        """
        device = self.model.device

        # Tokenize prompts only (no padding yet — we need prompt lengths)
        prompt_encodings = self.tokenizer(
            prompts, add_special_tokens=True, return_tensors=None, padding=False
        )
        prompt_ids_list = prompt_encodings["input_ids"]  # list of lists

        # Build full sequences: prompt + answer tokens; record prompt_end per sample
        full_ids_list = []
        prompt_ends = []  # index of last prompt token (0-based) per sample
        for p_ids, a_ids in zip(prompt_ids_list, answer_token_ids):
            full_ids_list.append(p_ids + list(a_ids))
            prompt_ends.append(len(p_ids) - 1)

        # Pad to longest (left-pad to match tokenizer.padding_side="left")
        max_len = max(len(s) for s in full_ids_list)
        pad_id = self.tokenizer.pad_token_id

        padded_ids = []
        attn_masks = []
        padded_prompt_ends = []  # prompt_end index after left-padding
        for ids, pe in zip(full_ids_list, prompt_ends):
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attn_masks.append([0] * pad_len + [1] * len(ids))
            padded_prompt_ends.append(pad_len + pe)  # shift by padding offset

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)
        prompt_end_idx = torch.tensor(padded_prompt_ends, dtype=torch.long, device=device)  # (B,)

        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}

        def forward_fn():
            return self.model(**tokens, return_dict=True, output_hidden_states=False, use_cache=False).logits

        if diff_matrices is not None:
            # Steering hook fires at prompt_end_idx positions only
            full_logits = self._apply_diff_hooks(
                diff_matrices, forward_fn, last_indices=prompt_end_idx, tail_len=1
            )  # (B, L, V)
        else:
            full_logits = forward_fn()  # (B, L, V)

        full_logits = full_logits.detach().cpu().to(torch.float32)

        # Extract per-answer-token logits: position (prompt_end + k) predicts answer token k+1,
        # so the logit for answer token k is at position (prompt_end + k - 1) for k>=1,
        # and at (prompt_end) for k=0 (first answer token predicted by last prompt token).
        # In left-padded indexing: base = padded_prompt_ends[i]
        results = []
        for i, (pe, a_ids) in enumerate(zip(padded_prompt_ends, answer_token_ids)):
            n_ans = len(a_ids)
            # positions that predict each answer token: pe, pe+1, ..., pe+n_ans-1
            positions = [pe + k for k in range(n_ans)]
            ans_logits = full_logits[i, positions, :]  # (n_ans, V)
            results.append(ans_logits.numpy())

        return results

    def regenerate_rsn_lesion(
        self,
        prompts: list[str],
        rsn_indices_per_layer: list[list[int]],
    ):
        """
        Lesion RSNs per layer and return last-token logits.
        """
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(self.model.device)
        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(
                **tokens,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False,
            ).logits  # shape: (B, L, V)

        full_logits = self._apply_rsn_lesion_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
        )  # shape: (B, L, V)

        B, L, V = full_logits.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=idx).squeeze(1)  # shape: (B, V)

        return last_logits.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def regenerate_rsn_complement(
        self,
        prompts: list[str],
        rsn_indices_per_layer: list[list[int]],
    ):
        """
        Complement Ablation:
        Keep only RSN neurons; zero out all other neurons.
        Return last-token logits for each prompt.
        """
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        ).to(self.model.device)

        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(
                **tokens,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False,
            ).logits  # shape: (B, L, V)

        full_logits = self._apply_rsn_complement_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
        )  # shape: (B, L, V)

        B, L, V = full_logits.shape
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=gather_idx).squeeze(1)

        return last_logits.detach().cpu().to(torch.float32).numpy()

    # ───────────────────── Generate answer ───────────────────── #
    @torch.no_grad()
    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Single-prompt, bs=1, padding=False generation. Use this when caller
        has its own forward hooks attached (tracker, closed-loop controller).

        Returns the generated text (prompt stripped, special tokens skipped).
        """
        do_sample = temperature > 0
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)
        prompt_len = input_ids.shape[1]

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=True,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(
        self,
        inputs: list[str],
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
        batch_size: int = 1,
        stop_strings: list[str] = None,
    ) -> list[str]:
        """
        Generate responses for a batch of input prompts.
        batch_size > 1 enables true batched inference for speedup.

        stop_strings: optional HF `generate(stop_strings=...)` markers. Default
            None keeps every existing caller byte-identical. It exists because
            `regenerate` accepted stop_strings and this did not, so an alpha=0
            cell (which takes this path, registering no hook) ran with NO stop
            marker while its own +-4 cells ran with one -- the two arms of the
            same experiment had different generation boundaries. Same caveat as
            regenerate: HF halts on the marker appearing ANYWHERE, so never use
            a marker that the prompt itself contains.
        """
        do_sample = temperature > 0
        top_p_val = top_p if do_sample else None
        temperature_val = temperature if do_sample else None

        results = []
        n_batches = (len(inputs) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(inputs), batch_size),
                      total=n_batches, desc="generate", unit="batch",
                      disable=n_batches <= 1):
            batch = inputs[i : i + batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokens.input_ids.to(self.model.device)
            attention_mask = tokens.attention_mask.to(self.model.device)
            prompt_len = input_ids.shape[1]

            gen_kwargs = dict(
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature_val,
                use_cache=True,
                top_p=top_p_val,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if stop_strings:
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = self.tokenizer

            output_ids = self.model.generate(input_ids, **gen_kwargs)
            for seq in output_ids:
                gen_ids = seq[prompt_len:]
                text = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                results.append(text.strip())

        return results

    @torch.no_grad()
    def generate_diffusion_llada(
        self,
        inputs: list[str],
        max_new_tokens: int = 4,
        steps: int = 50,
        block_len: int = 32,
        temperature: float = 0.0,
        guidance: float = 0.0,
    ) -> list[str]:
        """
        Use LLaDA's built-in diffusion sampling instead of the HF autoregressive generate method.
        """
        results = []

        for prompt in inputs:
            tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            full_ids = diffusion_generate(
                model=self.model,
                prompt_ids=tok.input_ids,
                gen_len=max_new_tokens,
                steps=steps,
                block_len=block_len,
                temperature=temperature,
                cfg_scale=guidance,
                remask="low_confidence",
            )
            gen_ids = full_ids[0, tok.input_ids.shape[1] :]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        return results

    @torch.no_grad()
    def generate_diffusion_dream(
        self,
        inputs: list[str],
        max_new_tokens: int = 4,
        steps: int = 50,
        temperature: float = 0.0,
        top_p: float = 0,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        output_history: bool = False,
        return_dict: bool = False,
    ) -> list[str]:
        """
        Dream-org/Dream-v0-Instruct-7B
        """
        results = []
        for prompt in inputs:
            toks = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

            out = self.model.diffusion_generate(
                toks.input_ids,
                attention_mask=toks.attention_mask,
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                output_history=output_history,
                return_dict_in_generate=return_dict,
            )

            seqs = out.sequences if return_dict else out
            gen_ids = seqs[0, toks.input_ids.shape[1] :]
            text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
            results.append(text)
        return results

    def steering_fire_count(self, reset: bool = False) -> int:
        """Observed steering injections since the last reset, in SITES.

        Attests that steering FIRED, as opposed to that it was configured —
        the config field and this counter can disagree only if there is a bug,
        which is the point of having both.

        A SITE is one (steered layer, sequence, token position) that actually
        received a non-zero add. Deliberately NOT "hook calls": diff_matrices
        is full-length with zero rows outside the steered band, and a batch of
        K candidates is ONE forward, so counting calls reports the same number
        for every layer count and every K — it could not distinguish 10 steered
        layers from 32, or K=4 from K=5. With L steered layers:
            regenerate(prefill_only, tail_len=t, B prompts) -> L * B * t
            regenerate_logits_teacher_forcing(K prompts)    -> L * K
        """
        n = self._steering_fire_count
        if reset:
            self._steering_fire_count = 0
        return n

    @torch.no_grad()
    def regenerate(
        self,
        inputs: list[str],
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
        diff_matrices: list[np.ndarray] = None,
        prefill_only: bool = True,
        batch_size: int = 1,
        stop_strings: list[str] = None,
        prefill_tail_len: int = 1,
    ) -> list[str]:
        """
        Generate text by modifying hidden states of each layer using diff_matrices.

        Args:
            prefill_only: If True (default), only apply intervention during prompt processing (prefill).
                         If False, apply intervention to every generation step (legacy behavior).
            batch_size: Number of prompts per forward pass (prefill_only=True only).
            stop_strings: Optional list of strings that halt generation as soon as
                         any is produced (HF `generate(stop_strings=...)`). Default
                         None preserves existing callers' behavior exactly. Only
                         honored on the prefill_only path (the only branch that
                         drives `model.generate` directly).
            prefill_tail_len: number of trailing prompt tokens to inject into during
                         prefill. Default 1 = original last-token-only behaviour.
                         >1 injects the last N prompt tokens (CGT --inject_turn).
                         Only honored on the prefill_only path.
        """
        if diff_matrices is None:
            raise ValueError("The difference matrices are not loaded. Please provide `diff_matrices` during method call.")

        if not prefill_only:
            if prefill_tail_len != 1:
                raise ValueError("prefill_tail_len>1 is only supported with prefill_only=True.")
            # Legacy behavior: hooks active during entire generation
            def forward_fn():
                return self.generate(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
            results = self._apply_diff_hooks(diff_matrices, forward_fn)
            return results

        self._prefill_batch_size = batch_size
        return self._regenerate_prefill_only(
            inputs=inputs,
            diff_matrices=diff_matrices,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stop_strings=stop_strings,
            prefill_tail_len=prefill_tail_len,
        )

    @torch.no_grad()
    def _regenerate_prefill_only(
        self,
        inputs: list[str],
        diff_matrices: list[np.ndarray],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        stop_strings: list[str] = None,
        prefill_tail_len: int = 1,
    ) -> list[str]:
        """
        Apply intervention only during prefill (prompt processing), not during generation.

        Strategy: Use sequence length to detect prefill vs decode.
        - Prefill: L > 1 (processing entire prompt)
        - Decode: L == 1 (processing single new token at a time)

        prefill_tail_len: how many trailing prompt tokens to inject into. Default 1
        keeps the historical behaviour (only the very last token, e.g. GSM8K/Bandit).
        Set >1 to inject into the last N prompt tokens — used by CGT --inject_turn to
        steer the whole final user turn (≈ the round's new board info) instead of a
        single token swamped by the multi-turn history. Clamped to L-1 so it never
        reaches into earlier turns past the current prefill block.
        """
        decoder_layers = self._find_decoder_layers()
        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"diff_matrices length ({len(diff_matrices)}) != layers ({len(decoder_layers)})"
            )

        do_sample = temperature > 0
        top_p_val = top_p if do_sample else None
        temperature_val = temperature if do_sample else None
        n_tail = max(int(prefill_tail_len), 1)

        # Create conditional hooks that only fire during prefill (L > 1)
        def create_prefill_hook(diff_matrix):
            # See _apply_diff_hooks: zero rows outside the steered band must not
            # count as injections. Evaluated once per layer, not per forward.
            _layer_is_steered = bool(np.any(np.asarray(diff_matrix) != 0))

            def hook(_module, _input, output):
                if isinstance(output, tuple):
                    hs = output[0]  # [B, L, H]
                else:
                    hs = output

                B, L, H = hs.shape

                # Only intervene if L > 1 (prefill stage)
                # During decode, L == 1 (single token), so skip intervention
                if L <= 1:
                    return output

                # Prefill: add diff to the last n_tail tokens of each sequence
                # (n_tail==1 reproduces the original last-token-only behaviour).
                # Clamp to L-1 so injection stays within this prefill block.
                n = min(n_tail, L - 1) if L > 1 else 1
                diff_t = torch.as_tensor(diff_matrix, device=hs.device, dtype=hs.dtype)
                if diff_t.ndim == 1:
                    diff_t = diff_t.unsqueeze(0)  # [1, H]
                diff_t = diff_t.expand(B, -1)  # [B, H]

                hs[:, -n:, :] += diff_t.unsqueeze(1)  # broadcast over the n tail positions

                # Observed injection counter, in SITES: B sequences x n tail
                # positions, and only on layers whose diff is non-zero. Same
                # unit as the _apply_diff_hooks counter so the two stages of a
                # pv6 episode are directly comparable. Nothing depends on it.
                if _layer_is_steered:
                    self._steering_fire_count += B * n

                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                else:
                    return hs
            return hook

        # Register hooks once (shared across all batches)
        hooks = []
        for layer, diff_mtx in zip(decoder_layers, diff_matrices):
            h = layer.register_forward_hook(create_prefill_hook(diff_mtx))
            hooks.append(h)

        results = []
        try:
            n_batches = (len(inputs) + self._prefill_batch_size - 1) // self._prefill_batch_size
            # disable when there's a single batch — callers that pre-chunk and
            # loop (e.g. get_answer_regenerate_gsm8k.py) already show a tqdm, so
            # this avoids a redundant 1/1 bar flashing per call.
            for i in tqdm(range(0, len(inputs), self._prefill_batch_size),
                          total=n_batches, desc="regenerate", unit="batch",
                          disable=n_batches <= 1):
                batch = inputs[i : i + self._prefill_batch_size]
                tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = tokens.input_ids.to(self.model.device)
                attention_mask = tokens.attention_mask.to(self.model.device)
                prompt_len = input_ids.shape[1]

                gen_kwargs = dict(
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature_val,
                    use_cache=True,
                    top_p=top_p_val,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if stop_strings:
                    # HF needs the tokenizer to map stop strings → token ids.
                    gen_kwargs["stop_strings"] = stop_strings
                    gen_kwargs["tokenizer"] = self.tokenizer
                output_ids = self.model.generate(input_ids, **gen_kwargs)
                for seq in output_ids:
                    gen_ids = seq[prompt_len:]
                    text = self.tokenizer.decode(
                        gen_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )
                    results.append(text.strip())
        finally:
            for h in hooks:
                h.remove()

        return results

    @torch.no_grad()
    def replace_generate(
        self,
        inputs: list[str],
        replace_matrices: list[np.ndarray] = None,
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
        start: int = 0,
        end: int = None,
    ) -> list[str]:
        """
        Generate text by directly replacing the last token's hidden states
        for layers in [start, end) with 'replace_matrices'.
        """
        if replace_matrices is None:
            raise ValueError("The replacement matrices must be provided.")

        def forward_fn():
            return self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )

        outputs = self._apply_replace_hooks(
            replace_matrices=replace_matrices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        return outputs

    @torch.no_grad()
    def regenerate_index_lesion(
        self,
        inputs: list[str],
        neuron_indices: list[int],
        start: int = 0,
        end: int = None,
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate text while zeroing out specified neuron indices in [start, end).
        """

        def forward_fn():
            return self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )

        outputs = self._apply_index_lesion_hooks(
            neuron_indices=neuron_indices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        return outputs

    # ───────────────────── Hidden state extractors ───────────────────── #

    @torch.no_grad()
    def get_hidden_states_mdf(self, prompt: str, diff_matrices: list[np.ndarray], **kwargs):
        """
        Get hidden states under diff_matrices (mdf), using the same diff-hook
        mechanism as regenerate_logits.
        """
        formatted_prompt = prompt
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        outputs = self._apply_diff_hooks(diff_matrices, forward_fn)
        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors

        positions = {"pos1": seq_len - 1}
        results = []

        for pos_name, index in positions.items():
            if index is not None and isinstance(index, int) and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    token_vec = layer_hs[0, index, :].detach().cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)
        return results

    @torch.no_grad()
    def get_hidden_states_rpl(
        self,
        prompt: str,
        replace_matrices: list[np.ndarray],
        start: int = 0,
        end: int = None,
        **kwargs,
    ):
        """
        Get hidden states when replacing last token's hidden state for layers in [start, end).
        """
        formatted_prompt = prompt
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        outputs = self._apply_replace_hooks(
            replace_matrices=replace_matrices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors

        positions = {"pos1": seq_len - 1}
        results = []

        for pos_name, index in positions.items():
            if index is not None and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    token_vec = layer_hs[0, index, :].detach().cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)

        return results

    @torch.no_grad()
    def get_hidden_states(self, prompt: str, character: str = None, **kwargs):
        """
        Basic hidden state extractor (no editing).
        """
        tokens = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors
        seq_len = tokens.input_ids.shape[1]
        positions = {"pos1": seq_len - 1}  # extract last token position
        results = []

        for pos_name, index in positions.items():
            if index is not None and isinstance(index, int) and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    token_vec = layer_hs[0, index, :].cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)

        return results
