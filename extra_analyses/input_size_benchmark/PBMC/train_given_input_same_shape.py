#!/data/groups/vib.ai/stein.aerts/sdewin/software/miniforge3/envs/crested_1.5.0/bin/python

import crested
import anndata
import keras
from datetime import datetime
import argparse
import os

now = datetime.now()
timestamp = str(now.year)
timestamp += str(now.month)
timestamp += str(now.day)
timestamp += str(now.hour)
timestamp += str(now.minute)
timestamp += str(now.second)

BIGWIGS_FOLDER="../crested_data/Figure_3/pbmc/bw/"
REGION_FILE="../crested_data/Figure_3/pbmc/consensus_regions.bed"


def dilated_cnn_same_shape(
    seq_len: int,
    num_classes: int,
    first_conv_filters: int = 512,
    first_conv_filter_size: int = 5,
    first_conv_pool_size: int = 0,
    first_conv_activation: str = "gelu",
    first_conv_l2: float = 0.00001,
    first_conv_dropout: float = 0.1,
    n_dil_layers: int = 8,
    num_filters: int = 512,
    filter_size: int = 3,
    activation: str = "relu",
    output_activation: str = "softplus",
    l2: float = 0.00001,
    dropout: float = 0.1,
    batch_norm: bool = True,
    dense_bias: bool = True,
) -> keras.Model:
    """
    Construct a CNN using dilated convolutions.

    This architecture is based on the ChromBPNet model described in :cite:`Pampari_Bias_factorized_base-resolution_2023`.
    This was renamed to DilatedCNN to avoid confusion with the original ChromBPNet framework.

    Parameters
    ----------
    seq_len
        Width of the input region.
    num_classes
        Number of classes to predict.
    first_conv_filters
        Number of filters in the first convolutional layer.
    first_conv_filter_size
        Size of the kernel in the first convolutional layer.
    first_conv_pool_size
        Size of the pooling kernel in the first convolutional layer.
    first_conv_activation
        Activation function in the first convolutional layer.
    first_conv_l2
        L2 regularization for the first convolutional layer.
    first_conv_dropout
        Dropout rate for the first convolutional layer.
    n_dil_layers
        Number of dilated convolutional layers.
    num_filters
        Number of filters in the dilated convolutional layers.
    filter_size
        Size of the kernel in the dilated convolutional layers.
    activation
        Activation function in the dilated convolutional layers.
    output_activation
        Activation function for the output layer.
    l2
        L2 regularization for the dilated convolutional layers.
    dropout
        Dropout rate for the dilated convolutional layers.
    batch_norm
        Whether or not to use batch normalization.
    dense_bias
        Whether or not to add a bias to the dense layer.

    Returns
    -------
    A Keras model.
    """
    # Model
    inputs = keras.layers.Input(shape=(seq_len, 4), name="sequence")

    # Convolutional block without dilation
    x = keras.layers.Conv1D(
        filters=first_conv_filters,
        kernel_size=first_conv_filter_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(first_conv_l2),
        use_bias=False,
    )(inputs)
    x = keras.layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(x)
    x = keras.layers.Activation(first_conv_activation)(x)
    if first_conv_pool_size > 1:
        x = keras.layers.MaxPooling1D(pool_size=first_conv_pool_size, padding="same")(x)
    x = keras.layers.Dropout(first_conv_dropout)(x)

    # Dilated convolutions
    layer_names = [str(i) for i in range(1, n_dil_layers + 1)]

    for i in range(1, n_dil_layers + 1):
        conv_layer_name = f"bpnet_{layer_names[i - 1]}conv"
        conv_x = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=1,
            activation=None,
            padding="same", #<--- MODIFIED THIS
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
            use_bias=False,
            dilation_rate=2**i,
            name=conv_layer_name,
        )(x)
        if batch_norm:
            conv_x = keras.layers.BatchNormalization(
                momentum=0.9,
                gamma_initializer="ones",
                name=f"bpnet_{layer_names[i - 1]}bn",
            )(conv_x)
        if activation != "none":
            conv_x = keras.layers.Activation(
                activation, name=f"bpnet_{layer_names[i - 1]}activation"
            )(conv_x)

        x_len = keras.ops.shape(x)[1]
        conv_x_len = keras.ops.shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0  # for symmetric cropping

        x = keras.layers.Cropping1D(
            (x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i - 1]}crop"
        )(x)
        x = keras.layers.add([conv_x, x])
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"bpnet_{layer_names[i-1]}dropout")(
                x
            )

    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(
        units=num_classes,
        activation=output_activation,
        use_bias=dense_bias,
        name="dense_out",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def calc_output_width_after_conv(input_width: int, kernel_size: int, dilation_factor: int) -> int:
    k_eff = dilation_factor * (kernel_size - 1) + 1
    return input_width - k_eff + 1

def calc_max_number_of_dil_layers(input_width: int, kernel_size) -> int:
    i = 0
    while input_width > 0:
        input_width = calc_output_width_after_conv(input_width, kernel_size, 2**(i + 1))
        i += 1
    return i - 1

def preprocess(input_size: int) -> anndata.AnnData:
    adata = crested.import_bigwigs(
        bigwigs_folder=BIGWIGS_FOLDER,
        regions_file=REGION_FILE,
        target_region_width=1000,
        target="mean",
    )
    crested.pp.train_val_test_split(adata, strategy="chr", val_chroms=["chr8", "chr10"], test_chroms=["chr9", "chr18"])
    print(f"CHANGING REGIONS WIDTH TO: {input_size}")
    crested.pp.change_regions_width(adata=adata, width=input_size)
    crested.pp.normalize_peaks(adata, top_k_percent = 0.03)
    return adata


def train(adata: anndata.AnnData, input_size: int):
    n_dil_layers = 8
    print(f"Setting up model training for input size: {input_size} using {n_dil_layers} dilation layers")
    datamodule = crested.tl.data.AnnDataModule(
        adata=adata,
        batch_size=128,
        max_stochastic_shift=3,
        always_reverse_complement=True
    )

    model_architecture = dilated_cnn_same_shape(
        seq_len=input_size,
        num_classes=7,
        n_dil_layers=n_dil_layers
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = crested.tl.losses.CosineMSELogLoss(max_weight=100)
    metrics = [
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.CosineSimilarity(axis=1),
                crested.tl.metrics.PearsonCorrelation(),
                crested.tl.metrics.ConcordanceCorrelationCoefficient(),
                crested.tl.metrics.PearsonCorrelationLog(),
                crested.tl.metrics.ZeroPenaltyMetric(),
    ]

    config = crested.tl.TaskConfig(optimizer=optimizer, loss=loss, metrics=metrics)

    trainer = crested.tl.Crested(
        data = datamodule,
        model = model_architecture,
        config = config,
        project_name = "PBMC_INPUT_SIZE",
        run_name = f"input_{input_size}_{timestamp}",
        logger = 'wandb',
    )

    trainer.fit(epochs=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_width", required = True, type = int)

    args = parser.parse_args()
    input_size = args.input_width

    # change directory because import_bigwigs
    # creates a temp file, so otherwise not possible to train
    # multiple models in parallel
    os.makedirs(f"input_same_shape_{input_size}_{timestamp}")
    os.chdir(f"input_same_shape_{input_size}_{timestamp}")

    GENOME_FA="../../../../../../../../resources/hg38/hg38.fa"
    GENOME_CHROM_SIZES="../../../../../../../../resources/hg38/hg38.chrom.sizes"


    crested.register_genome(
        crested.Genome(
            fasta=GENOME_FA,
            chrom_sizes=GENOME_CHROM_SIZES
        )
    )

    adata = preprocess(input_size)
    adata.write_h5ad(f"adata_input_{input_size}_{timestamp}.h5ad")

    train(adata, input_size)


if __name__ == "__main__":
    main()