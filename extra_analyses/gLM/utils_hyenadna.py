import json
import os
from collections.abc import Sequence
from pathlib import Path
from pickle import UnpicklingError
from typing import Optional, Union


import torch
from standalone_hyenadna import HyenaDNAModel
from utils_finetuning import BaseScalarModel, BaseTokenizerAnnDataset, BaseTokenizerAnnDataModule
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


# ---- Dataloader utilities ----

# TokenizerData* objects, with proper interfacing to HyenaDNA's tokenizer
class HyenaDNATokenizerAnnDataset(BaseTokenizerAnnDataset):
    def _get_tokens(self, seq):
        x = self.tokenizer(seq, return_tensors="pt")["input_ids"]
        return x

class HyenaDNATokenizerAnnDataModule(BaseTokenizerAnnDataModule):
    def _create_dataset(self, *args, **kwargs):
        """Dummy function to let you define your own dataset class without repeating a lot of boilerplate"""
        return HyenaDNATokenizerAnnDataset(*args, **kwargs)

# ---- Tokenizer utilities ----

# Rework of HyenaDNA's CharacterTokenizer which is broken on my transformers version (4.54.1)
# Their tokenizer doesn't implement get_vocab() (which is called in super().__init__()) -> fixing here
# Presumably it's built for another older version of the transformers library?
# To implement that: moving self._vocab_str_to_int before super().__init__() so that we have a vocab to return in get_vocab()
class CharacterTokenizerv2(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Parameters
        ----------
        characters
            List of desired characters, i.e. ['A', 'C', 'T', 'G', 'N']. Any character which is not 
            included in this list will be replaced by a special token called [UNK] with id=6. 
            Following is a list of all of the special tokens with their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                    an id (starting at 7) will be assigned to each character.
        model_max_length
            Model maximum sequence length. Won't automatically pad to this, but will cut off if longer than it.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    def get_vocab(self):
        return self._vocab_str_to_int

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CharacterTokenizerv2":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

# ---- Model utilities ----

def load_weights(scratch_dict, pretrained_dict):
    """Loads pretrained (backbone only) weights into the scratch state dict. 
    Editor's note: had to rework this to also add .layer. for it to properly return weights that could be loaded. Also, this function doesn't load weights but just returns an adjusted state dict.

    scratch_dict: dict, a state dict from a newly initialized HyenaDNA model
    pretrained_dict: dict, a state dict from the pretrained ckpt

    return:
    dict, a state dict with the pretrained weights loaded (head is scratch)

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it
    """
    
    # need to do some state dict "surgery"
    for key in scratch_dict:
        orig_key = key
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key = 'model.' + key
        # need to add an extra ".layer" in key
        key = key.replace('mixer.', 'mixer.layer.')
        key = key.replace('mlp.', 'mlp.layer.')
        # breakpoint()

        try:
            scratch_dict[orig_key] = pretrained_dict[key]
        except KeyError as e:
            raise KeyError(f"Couldn't find {key} in pretrained_dict. Attempting to find equivalent of {orig_key} from scratch_dict.") from e

    # scratch_dict has been updated
    return scratch_dict

# Note: builds the base model, not the model we'll use to train. Pass the result of this to HyenaDNAScalarModel(base_model=base_model).
def build_hyenadna_model(
    model_path,
    config=None,
    device='cpu',
    use_head=False,
    n_classes=2,
):
    """
    Rework of `HyenaDNAPreTrainedModel` from https://github.com/HazyResearch/hyena-dna/blob/main/huggingface.py

    We don't care about the huggingface layout, just want the model returning embeddings.
    """
    # first check if it is a local path
    if os.path.isdir(model_path):
        if config is None:
            config = json.load(open(os.path.join(model_path, 'config.json')))
    else:
        raise FileNotFoundError("Please clone the HyenaDNA model of choice to the path supplied: git lfs clone https://huggingface.co/LongSafari/{model_name}. Make sure git-lfs is installed, or you'll get 'Unsupported operand 118' errors when loading the model.")

    scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
    try:
        loaded_ckpt = torch.load(
            os.path.join(model_path, 'weights.ckpt'),
            map_location=torch.device(device),
            weights_only=False
        )
    except UnpicklingError as e:
        raise UnpicklingError("You ran into an unpickling error, meaning the weights file is likely corrupt due to lack of git-lfs when downloading from huggingface (see https://github.com/pytorch/pytorch/issues/150998). Install git-lfs or download from huggingface manually. Note that even with git-lfs installed, just using git clone can still result in corruption - git lfs clone worked for me.") from e

    # grab state dict from both and load weights
    state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'])

    # scratch model has now been updated
    scratch_model.load_state_dict(state_dict)
    print("Loaded pretrained weights ok!")
    return scratch_model

class HyenaDNAScalarModel(BaseScalarModel):
     def model_pass(self, X, **kwargs):
        pred = self.base_model(X, **kwargs)
        # pred = torch.flatten(pred, start_dim = 1)
        pred = torch.mean(pred, axis = -2)
        pred = self.head(pred, **kwargs)
        return pred

