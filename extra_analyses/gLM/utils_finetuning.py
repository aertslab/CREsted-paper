from collections.abc import Sequence

import lightning
import numpy as np
import torch

import crested

# ---- Dataloader utilities ----

class BaseTokenizerAnnDataset(crested.tl.data.AnnDataset):
    """Base version of a tokenizer-compatible AnnDataset object. Inherit and overwrite _get_tokens() to properly use the tokenizing logic of your tokenizer."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        """Return sequence and target for a given index. AnnDataset's __getitem__ except one-hot encoding is replaced by tokenizing."""
        augmented_index = self.index_manager.augmented_indices[idx]
        original_index = self.index_manager.augmented_indices_map[augmented_index]
        # stochastic shift
        if self.max_stochastic_shift > 0:
            shift = np.random.randint(
                -self.max_stochastic_shift, self.max_stochastic_shift + 1
            )
        else:
            shift = 0

        # Get sequence
        x = self.sequence_loader.get_sequence(
            augmented_index, stranded=True, shift=shift
        )

        # random reverse complement (always_reverse_complement is done in the sequence loader)
        if self.random_reverse_complement and np.random.rand() < 0.5:
            x = self.sequence_loader._reverse_complement(x)

        # tokenize sequence and convert to numpy array
        x = self._get_tokens(x) # Only adjusted line -> used to be one-hot encoding
        y = self._get_target(original_index)

        return x, y

    def _get_tokens(x):
        """
        Function that turns a (consistent-length) nucleotide string into a consistent-length tokenized tensor.
        
        Simple example with padding would be the following:
        `return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length = 2048)["input_ids"]`
        """
        raise NotImplementedError

class TokenizerAnnDataLoader(crested.tl.data.AnnDataLoader):
    """Tokenizer-compatible version of AnnDataLoader."""

    def _collate_fn(self, batch):
        """Collate function to move tensors to the specified device if backend is torch."""
        inputs, targets = zip(*batch)
        # Ignore conversion to tensors for inputs since they're already output as tensors with batch dims
        # If inputs is a tuple, process like it, else just leave as is
        if isinstance(inputs[0], Sequence):
            inputs = tuple(torch.cat(subinputs, axis=0).to(self.device) for subinputs in zip(*inputs))
        else:
            inputs = torch.cat(inputs, axis = 0).to(self.device)
        targets = torch.stack([torch.tensor(target) for target in targets]).to(self.device)
        return inputs, targets


class BaseTokenizerAnnDataModule(crested.tl.data.AnnDataModule):
    """
    AnnDataModule except AnnDataModule.setup is changed to let you provide your own AnnDataset definition through _create_dataset(), and it now passes the tokenizer.

    To use this object, inherit it and overwrite _create_dataset() with your inherited version of BaseTokenizerAnnDataset of choice.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def _create_dataset(self, *args, **kwargs):
        """Dummy function to let you define your own dataset class without repeating a lot of boilerplate"""
        # return TokenizerAnnDataset(*args, **kwargs)
        return NotImplementedError("Define _create_dataset() for TokenizerAnnDataModule by overwriting the method with 'return YourAnnDataset(*args, **kwargs)'")

    def setup(self, stage: str) -> None:
        """
        Set up the Anndatasets for a given stage.

        Generates the train, val, test or predict dataset based on the provided stage.
        Should always be called before accessing the dataloaders.
        Generally you don't need to call this directly, as this is called inside the `tl.Crested` trainer class.

        Parameters
        ----------
        stage
            Stage for which to setup the dataloader. Either 'fit', 'test' or 'predict'.
        """
        if stage == "fit":
            self.train_dataset = self._create_dataset(
                anndata=self.adata,
                genome=self.genome,
                split="train",
                in_memory=self.in_memory,
                always_reverse_complement=self.always_reverse_complement,
                random_reverse_complement=self.random_reverse_complement,
                max_stochastic_shift=self.max_stochastic_shift,
                deterministic_shift=self.deterministic_shift,
                tokenizer=self.tokenizer,
            )
            self.val_dataset = self._create_dataset(
                anndata=self.adata,
                genome=self.genome,
                split="val",
                in_memory=self.in_memory,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
                tokenizer=self.tokenizer,
            )
        elif stage == "test":
            self.test_dataset = self._create_dataset(
                anndata=self.adata,
                genome=self.genome,
                split="test",
                in_memory=False,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
                tokenizer=self.tokenizer,
            )
        elif stage == "predict":
            self.predict_dataset = self._create_dataset(
                anndata=self.adata,
                genome=self.genome,
                split=None,
                in_memory=False,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
                tokenizer=self.tokenizer,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def train_dataloader(self):
        """:obj:`crested.tl.data.AnnDataLoader`: Training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Run setup('fit') first.")
        return TokenizerAnnDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_remainder=False,
        )

    @property
    def val_dataloader(self):
        """:obj:`crested.tl.data.AnnDataLoader`: Validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set. Run setup('fit') first.")
        return TokenizerAnnDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    @property
    def test_dataloader(self):
        """:obj:`crested.tl.data.AnnDataLoader`: Test dataloader."""
        if self.test_dataset is None:
            raise ValueError("test_dataset is not set. Run setup('test') first.")
        return TokenizerAnnDataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    @property
    def predict_dataloader(self):
        """:obj:`crested.tl.data.AnnDataLoader`: Prediction dataloader."""
        if self.predict_dataset is None:
            raise ValueError("predict_dataset is not set. Run setup('predict') first.")
        return TokenizerAnnDataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

# ---- Model utilities ----

# No Keras, just PyTorch Lightning
class BaseScalarModel(lightning.LightningModule):
    """
    Basic Pytorch Lightning module object, wrapping a base model with a linear+softplus head.

    To use this, inherit it and overwite model_pass() with your model forward pass logic.
    """

    def __init__(self, base_model, loss, lr, emb_shape, num_classes, *args, **kwargs):
        """Basic Pytorch Lightning module object, wrapping a base model with a linear+softplus head.

        Parameters
        ----------
        base_model
            The base model that turns tokens into embeddings. Expected to be a PyTorch module.
        loss
            The loss function/object, anything that returns a loss tensor when called as loss(y_pred = pred, y_true = Y).
        lr
            Learning rate to train the model with.
        emb_shape
            Shape of the inputs to the head (i.e. the embeddings, presumably averaged or pooled or w/e you do in model_pass()).
        num_classes
            Shape of the outputs from the head (i.e. the number of classes you're predicting)
        *args, *kwargs
            Any arguments to be passed to lightning.LightningModule.
        """
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features = emb_shape, out_features = num_classes),
            torch.nn.Softplus()
        )
        self.loss = loss
        self.lr = lr
        self.save_hyperparameters(ignore=['base_model'])

    def model_pass(self, X, **kwargs):
        """Basic model step to be implemented yourself when inheriting.

        Parameters
        ----------
        X
            Batched inputs of the model, as provided by the DataLoader.
        *args, **kwargs
            Arguments to pass on to the layer calls.
        """
        raise NotImplementedError

    def training_step(self, inputs, batch_idx=None, **kwargs):
        """A simple training step of model pass, loss calculation, and metric calculation."""
        X, Y = inputs
        pred = self.model_pass(X, **kwargs)
        loss = self.loss(y_pred = pred, y_true = Y)
        self.log('train_loss', loss)
        self.log('train_pearson', pearson(y_pred = pred, y_true = Y))
        self.log('train_mse', torch.nn.functional.mse_loss(input = pred, target = Y))
        return loss

    def validation_step(self, inputs, batch_idx=None, **kwargs):
        """A simple validation step of model pass, loss calculation, and metric calculation."""
        X, Y = inputs
        pred = self.model_pass(X, **kwargs)
        loss = self.loss(y_pred = pred, y_true = Y)
        self.log('val_loss', loss)
        self.log('val_pearson', pearson(y_pred = pred, y_true = Y))
        self.log('val_mse', torch.nn.functional.mse_loss(input = pred, target = Y))

    def test_step(self, inputs, batch_idx=None, **kwargs):
        """A simple test step of model pass, loss calculation, and metric calculation."""
        X, Y = inputs
        pred = self.model_pass(X, **kwargs)
        loss = self.loss(y_pred = pred, y_true = Y)
        self.log('test_loss', loss)
        self.log('test_pearson', pearson(y_pred = pred, y_true = Y))
        self.log('test_mse', torch.nn.functional.mse_loss(input = pred, target = Y))

    def configure_optimizers(self):
        """Obligatory optimizer creation function."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def pearson(y_true, y_pred):
    """Function to calculate Pearson correlation across all classes, based on crested.tl.metrics.PearsonCorrelation()."""
    count = y_true.numel()
    numerator = count * torch.sum(y_true * y_pred) - torch.sum(y_true) * torch.sum(y_pred)
    denominator = torch.sqrt(
        (count * torch.sum(torch.square(y_true)) - torch.square(torch.sum(y_true)))
        * (count * torch.sum(torch.square(y_pred)) - torch.square(torch.sum(y_pred)))
    )

    return numerator / (denominator + 1e-10)
