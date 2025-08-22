import torch
from utils_finetuning import BaseScalarModel, BaseTokenizerAnnDataset, BaseTokenizerAnnDataModule

# TokenizerData* objects, with proper interfacing to Nucleotide Transformer's tokenizer
class NucTransTokenizerAnnDataset(BaseTokenizerAnnDataset):
    def _get_tokens(self, seq):
        tokens = self.tokenizer(seq, return_tensors="pt", padding="max_length", max_length = 2048)["input_ids"]
        mask = tokens != self.tokenizer.pad_token_id
        return (tokens, mask)

class NucTransTokenizerAnnDataModule(BaseTokenizerAnnDataModule):
    def _create_dataset(self, *args, **kwargs):
        """Dummy function to let you define your own dataset class without repeating a lot of boilerplate"""
        return NucTransTokenizerAnnDataset(*args, **kwargs)

class NucTransScalarModel(BaseScalarModel):
    def model_pass(self, X, **kwargs):
        tokens, mask = X
        # Get embeddings
        pred = self.base_model(
            tokens,
            attention_mask=mask,
            encoder_attention_mask=mask,
            **kwargs
        )['last_hidden_state'] # requires base AutoModel, not AutoModelForMaskedLM or anything like that!
        
        # Get mean embedding over non-padding values
        mask = torch.unsqueeze(mask, dim=-1)
        pred = torch.sum(mask*pred, axis=-2)/torch.sum(mask, axis=1)
        
        # Apply output head to region embedding
        pred = self.head(pred, **kwargs)
        return pred
    
