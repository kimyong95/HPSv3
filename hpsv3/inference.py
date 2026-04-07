
import os
import types
from collections.abc import Mapping
import torch
import huggingface_hub
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from .dataset.utils import process_vision_info
from .dataset.data_collator_qwen import prompt_with_special_token, prompt_without_special_token, INSTRUCTION
from .utils.parser import ModelConfig, PEFTLoraConfig, DataConfig
from .train import create_model_and_processor
from pathlib import Path

_MODEL_CONFIG_PATH = Path(__file__).parent / f"config/"

class HPSv3RewardInferencer():
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda', differentiable=False):
        if config_path is None:
                config_path = os.path.join(_MODEL_CONFIG_PATH, 'HPSv3_7B.yaml')

        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download("MizzenAI/HPSv3", 'HPSv3.safetensors', repo_type='model')

        args = OmegaConf.to_container(OmegaConf.load(config_path))
        args.pop('deepspeed', None)

        parser = HfArgumentParser((DataConfig, ModelConfig, PEFTLoraConfig))
        data_config, model_config, peft_lora_config = parser.parse_dict(args, allow_extra_keys=True)

        # Minimal namespace with only the fields create_model_and_processor needs
        training_args = types.SimpleNamespace(
            output_dir=os.path.join(
                args.get('output_dir', 'output_models'),
                str(config_path).split("/")[-1].split(".")[0],
            ),
            disable_flash_attn2=args.get('disable_flash_attn2', False),
            bf16=args.get('bf16', False),
            fp16=args.get('fp16', False),
        )

        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        if checkpoint_path.endswith('.safetensors'):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path , map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = self._adapt_qwen2vl_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)
        self.data_config = data_config

    @staticmethod
    def _adapt_qwen2vl_state_dict(model, state_dict):
        adapt_state_dict = {}
        reverse_mappings = {
            "model.visual.": "visual.",
            "model.language_model.layers.": "model.layers.",
            "model.language_model.norm.": "model.norm.",
            "model.language_model.embed_tokens.": "model.embed_tokens.",
            "model.language_model.lm_head.": "lm_head."
        }
        
        for exp_key in model.state_dict().keys():
            # 1. Direct match
            if exp_key in state_dict:
                adapt_state_dict[exp_key] = state_dict[exp_key]
                
            # 2. Check for possible old key
            elif (old_key := next((exp_key.replace(n, o, 1) for n, o in reverse_mappings.items() if exp_key.startswith(n)), None)) and old_key in state_dict:
                adapt_state_dict[exp_key] = state_dict[old_key]
            
            # 3. If still not found, raise an error
            else:
                raise RuntimeError(f"Strict Error: Missing key '{exp_key}'. Not found in old or new format.")
                
        return adapt_state_dict

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def reward(self, prompts, image_paths):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        return rewards
    
    


if __name__ == "__main__":
    config_path = 'config/inference/HPSv3_7B.yaml'
    checkpoint_path = 'checkpoints/HPSv3_7B.pth'
    device = 'cuda'
    dtype = torch.bfloat16
    inferencer = HPSv3RewardInferencer(config_path, checkpoint_path, device=device)

    image_paths = [
        "assets/example1.png",
        "assets/example2.png"
    ]
    prompts = [
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
    ]
    rewards = inferencer.reward(image_paths, prompts)
    print(rewards[0][0].item()) # miu and sigma. we select miu as the final output
    print(rewards[1][0].item())
