import crested
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from tqdm import tqdm

def get_regions_from_dataframe(df: pd.DataFrame):
    """
    Iterates through a DataFrame with columns: chrom, start, end.
    
    Parameters:
    - df: DataFrame with columns 'chrom', 'start', 'end' and potentially other metadata.
    
    Yields:
    - Tuples of (chrom, start, end) for each region in the DataFrame.
    """
    for idx, row in df.iterrows():
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])
        yield chrom, start, end
            
def get_hot_encoding_table(
    alphabet: str = 'ACGT',
    neutral_alphabet: str = 'N',
    neutral_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Get hot encoding table to encode a DNA sequence to a numpy array with shape
    (len(sequence), len(alphabet)).
    """
    # Create a dictionary mapping nucleotides to one-hot encodings
    nucleotide_to_index = {nuc: idx for idx, nuc in enumerate(alphabet)}
    
    # Initialize the hot encoding table to size (len(alphabet) + neutral base, len(alphabet)).
    hot_encoding_table = np.zeros((len(nucleotide_to_index) + 1, len(alphabet)), dtype=dtype)
    
    # Set the one-hot encodings for nucleotides in the alphabet
    for nucleotide, index in nucleotide_to_index.items():
        hot_encoding_table[index, index] = 1.0
    
    # Set the neutral value for the ambiguous base 'N'
    neutral_index = len(nucleotide_to_index)  # The next index after A, C, G, T
    hot_encoding_table[neutral_index, :] = neutral_value
    
    return hot_encoding_table
    
# Modify the regions_to_hot_encoding function to accept a DataFrame
def regions_to_hot_encoding(df: pd.DataFrame, genomic_pysam: pysam.FastaFile, hot_encoding_table: np.ndarray):
    """
    Encode the sequence associated with each region in the DataFrame
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).
    
    Parameters:
    - df: DataFrame containing regions with 'chrom', 'start', 'end' columns.
    - genomic_pysam: pysam FastaFile object to fetch genomic sequences.
    - hot_encoding_table: Numpy array to convert nucleotides to hot-encoded arrays.
    """
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}  # Handles 'N' for ambiguous bases.

    for idx, row in df.iterrows():
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])

        # Fetch the sequence from the fasta file
        sequence = genomic_pysam.fetch(chrom, start, end).upper()

        # Convert sequence to indices using nucleotide_to_index mapping, handle unrecognized bases with 'N'.
        sequence_indices = np.array([nucleotide_to_index.get(base, 4) for base in sequence], dtype=np.uint8)

        # Hot encode the region sequence using the hot_encoding_table.
        hot_encoded = hot_encoding_table[sequence_indices]

        yield hot_encoded

def one_hot_encode_sequences(df, seq_len, pysam_mm, pysam_hg, hot_encoding_table):
    """
    Function to one-hot encode sequences for both mouse and human species and return the concatenated array.
    
    Parameters:
    - df: DataFrame containing regions with 'chrom', 'start', 'end', and 'species' columns.
    - seq_len: Length to which sequences will be padded/truncated.
    - pysam_mm: pysam FastaFile object for mouse genome.
    - pysam_hg: pysam FastaFile object for human genome.
    - hot_encoding_table: Table used for hot encoding the sequences.
    
    Returns:
    - Concatenated one-hot encoded array for mouse and human sequences.
    """
    # Initialize one-hot encoded arrays for mouse and human
    seqs_one_hot_mm = np.zeros((len(df[df['species'] == 'mouse']), seq_len, 4))
    seqs_one_hot_hg = np.zeros((len(df[df['species'] == 'human']), seq_len, 4))

    # Separate DataFrame rows based on species
    df_mouse = df[df['species'] == 'mouse']
    df_human = df[df['species'] == 'human']

    # One-hot encoding for mouse regions
    for i, hot_encoded_region in tqdm(enumerate(regions_to_hot_encoding(df_mouse, pysam_mm, hot_encoding_table)), total=len(df_mouse)):   
        padding = (seq_len - len(hot_encoded_region)) // 2
        seqs_one_hot_mm[i][padding:padding + len(hot_encoded_region)] = hot_encoded_region  

    # One-hot encoding for human regions
    for i, hot_encoded_region in tqdm(enumerate(regions_to_hot_encoding(df_human, pysam_hg, hot_encoding_table)), total=len(df_human)):   
        padding = (seq_len - len(hot_encoded_region)) // 2
        seqs_one_hot_hg[i][padding:padding + len(hot_encoded_region)] = hot_encoded_region  

    # Concatenate the mouse and human arrays
    seqs_one_hot = np.concatenate((seqs_one_hot_mm, seqs_one_hot_hg))

    return seqs_one_hot

def plot_prediction_heatmaps(df, prediction_dict, prediction_sources, figsize_scale=10, save_file=None):
    """
    Plots heatmaps comparing actual versus predicted cell types using multiple prediction methods.
    
    Parameters:
    - df: DataFrame containing actual data with 'target_ct' and 'enhancerID' columns.
    - prediction_dict: Dictionary of predictions with 'enhancerID' as the key and predictions as values.
    - prediction_sources: List of prediction methods to plot (keys in the prediction dictionary).
    - figsize_scale: Scaling factor for the figure size (default is 12 for each subplot).
    
    Returns:
    - None. Displays heatmaps comparing actual and predicted cell types for each method.
    """
    # Convert the predictions dictionary to a DataFrame with separate columns for each prediction type
    predictions_df = pd.DataFrame.from_dict(prediction_dict, orient='index').reset_index()
    predictions_df.rename(columns={'index': 'enhancerID'}, inplace=True)

    # Merge the predictions DataFrame with the actual DataFrame on enhancerID
    merged_df = pd.merge(df, predictions_df, on='enhancerID')

    # Set up the subplots
    nplots = len(prediction_sources)
    fig, axes = plt.subplots(1, nplots, figsize=(figsize_scale * nplots, 8))

    # If only one plot, ensure axes is a list for compatibility with iteration
    if nplots == 1:
        axes = [axes]

    # Iterate through each prediction source and create a heatmap subplot
    for i, prediction_source in enumerate(prediction_sources):
        # Create a contingency table (pivot table) of actual versus predicted cell types
        filtered_df = merged_df.copy()
        contingency_table = pd.crosstab(filtered_df['target_ct'], filtered_df[prediction_source])
        
        # Plot the heatmap in the corresponding subplot
        sns.heatmap(contingency_table, annot=True, cmap="magma", fmt="d", ax=axes[i])
        axes[i].set_title(f'Heatmap of Target vs. {prediction_source}')
        axes[i].set_xlabel('Predicted Cell Type')
        axes[i].set_ylabel('Actual Cell Type')

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(df_subset, prediction_dict, prediction_sources):
    """
    Calculate accuracy, precision, recall, and F1-score for each prediction source.
    
    Parameters:
    - df_subset: DataFrame containing the actual labels with 'target_ct' and 'enhancerID' columns.
    - prediction_dict: Dictionary with predictions for each 'enhancerID' for multiple prediction sources.
    - prediction_sources: List of prediction methods to evaluate.
    
    Returns:
    - performance_df: DataFrame containing accuracy, precision, recall, and F1-score for each prediction method.
    """
    # Convert the predictions dictionary to a DataFrame
    predictions_df = pd.DataFrame.from_dict(prediction_dict, orient='index').reset_index()
    predictions_df.rename(columns={'index': 'enhancerID'}, inplace=True)

    # Merge predictions with actual labels
    merged_df = pd.merge(df_subset, predictions_df, on='enhancerID')

    # Initialize a dictionary to store classification reports for each prediction method
    performance_reports = {}

    # Iterate through each prediction source and calculate performance metrics
    for prediction_source in prediction_sources:
        # Check if the prediction source exists in the DataFrame
        if prediction_source not in merged_df.columns:
            print(f"Prediction source '{prediction_source}' not found in the DataFrame.")
            continue

        # Filter out rows where predictions are missing
        filtered_df = merged_df.dropna(subset=[prediction_source, 'target_ct'])

        if filtered_df.empty:
            print(f"No valid data for prediction source '{prediction_source}'. Skipping...")
            continue

        # Calculate classification report and accuracy
        report = classification_report(filtered_df['target_ct'], filtered_df[prediction_source], output_dict=True)
        accuracy = accuracy_score(filtered_df['target_ct'], filtered_df[prediction_source])

        # Store the performance report for this prediction source
        performance_reports[prediction_source] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }

    # Convert performance reports to a DataFrame
    performance_df = pd.DataFrame(performance_reports).T

    return performance_df

def plot_performance_metrics(performance_df, figsize=(6,3)):
    """
    Plot a bar chart of the performance metrics across different prediction methods.
    
    Parameters:
    - performance_df: DataFrame containing accuracy, precision, recall, and F1-score for each prediction method.
    
    Returns:
    - None. Displays the plot.
    """
    # Plot the performance metrics as a bar chart
    performance_df.plot(kind='bar', figsize=figsize)
    plt.title('Performance Comparison Across Prediction Methods')
    plt.ylabel('Score')
    plt.xlabel('Prediction Method')
    plt.xticks(rotation=0)
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()

def calculate_specificity(arr: np.ndarray, scale: bool = False) -> np.ndarray:
    """
    Calculates the specificity of each element in a numpy array.
    For a 2D array, the specificity for each element is defined as its value divided by the sum of all elements in its row.
    For a 1D array, the specificity is each element divided by the sum of all elements.
    
    Parameters:
    - arr: A 1D or 2D numpy array of numerical values.
    
    Returns:
    - A numpy array containing the specificity scores for each element in the input array, 
      calculated per row for 2D arrays or per array for 1D arrays.
    """
    if arr.ndim == 1:
        total = np.sum(arr)
        if total == 0:
            total = 1e-6
        specificity_scores = (arr / total) * arr if scale else arr/total
    elif arr.ndim == 2:
        total_per_row = np.sum(arr, axis=1, keepdims=True)
        total_per_row[total_per_row == 0] = 1e-6
        specificity_scores = (arr / total_per_row) * arr if scale else arr/total_per_row
    else:
        raise ValueError("Input array must be 1D or 2D.")
    
    return specificity_scores

def calc_gini(targets: np.ndarray) -> np.ndarray:
    """Returns gini scores for the given targets"""

    def _gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = (
            array.flatten().clip(0, None) + 0.0000001
        )  # Ensure non-negative values and avoid zero
        array = np.sort(array)
        index = np.arange(1, array.size + 1)
        return (np.sum((2 * index - array.size - 1) * array)) / (
            array.size * np.sum(array)
        )

    gini_scores = np.zeros_like(targets)

    for region_idx in np.arange(0,targets.shape[0]):
        region_scores = targets[region_idx]
        max_idx = np.argmax(region_scores)
        gini_scores[region_idx, max_idx] = _gini(region_scores)
    gini_scores = np.max(gini_scores, axis=1)

    return gini_scores

def plot_bar_for_scores(index, classes, gt_values, scoring_arrays, scoring_labels=None, figsize=(25, 2)):
    """
    Function to plot bar charts for ground truth values and multiple scoring arrays.
    
    Parameters:
    - index: The index of the data to plot.
    - classes: List of class labels corresponding to the scores.
    - gt_values: Ground truth values array.
    - scoring_arrays: List of scoring arrays (e.g., spec_scores) to plot.
    - scoring_labels: List of labels for each scoring array (optional). Defaults to 'Score X' if not provided.
    - figsize: Tuple to define the figure size for each plot (default is (25, 3)).
    
    Returns:
    - None. Displays the plots.
    """
    # If no labels are provided, generate default labels for each scoring array
    if scoring_labels is None:
        scoring_labels = [f'Score {i+1}' for i in range(len(scoring_arrays))]
    
    # Plot the ground truth values
    plt.figure(figsize=figsize)
    plt.bar(classes, gt_values[index])
    plt.title('Ground Truth Values')
    plt.show()
    
    # Plot each of the scoring arrays
    for i, scores in enumerate(scoring_arrays):
        plt.figure(figsize=figsize)
        plt.bar(classes, scores[index])
        plt.title(scoring_labels[i])
        plt.show()

def plot_summary_roc_aupr_binary(models_data, file=''):
    plt.figure(figsize=(16,8))
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red','purple','red','cyan']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),'--','-.',':']
    
    # ROC plot
    plt.subplot(1, 2, 1)
    for model, color, linestyle in zip(models_data, colors, linestyles):
        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(model['y_true'].ravel(), model['y_score'].ravel())
        roc_auc_micro = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linestyle=linestyle,
             label=f'{model["name"]} (roc_auc = {roc_auc_micro:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on Binary Classification for enhancer activity')
    plt.legend(loc="lower right")
    
    # Precision-Recall plot
    plt.subplot(1, 2, 2)
    for model, color, linestyle in zip(models_data, colors, linestyles):
        precision, recall, _ = precision_recall_curve(model['y_true'].ravel(), model['y_score'].ravel())
        average_precision_micro = average_precision_score(model['y_true'], model['y_score'], average="micro")
        average_recall = np.mean(recall)

        plt.plot(recall, precision, color=color, linestyle=linestyle,
             label=f'{model["name"]} (AP = {average_precision_micro:.2f}, AR = {average_recall:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve on Binary Classification for enhancer activity')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    if (len(file)>0):
        plt.savefig(file, format='svg')
    plt.show()
    
def plot_summary_roc_aupr(models_data, file=''):
    plt.figure(figsize=(16,8))
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red','purple','red']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),'--','-.']
    
    # ROC plot
    plt.subplot(1, 2, 1)
    for model, color, linestyle in zip(models_data, colors, linestyles):
        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(model['y_true'].ravel(), model['y_score'].ravel())
        roc_auc_micro = auc(fpr, tpr)

        # Compute macro-average ROC area
         # Compute macro-average ROC area, excluding classes with no positive samples
        valid_indices = [i for i in range(model['y_true'].shape[1]) if np.sum(model['y_true'][:, i]) > 0]
        if valid_indices:
            roc_auc_macro = np.mean([auc(roc_curve(model['y_true'][:, i], model['y_score'][:, i])[0], roc_curve(model['y_true'][:, i], model['y_score'][:, i])[1]) for i in valid_indices])
        else:
            roc_auc_macro = np.nan  # Or handle as needed if no valid classes
        plt.plot(fpr, tpr, color=color, linestyle=linestyle,
             label=f'{model["name"]} (macro-average area = {roc_auc_macro:.2f}, micro-average area = {roc_auc_micro:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Summary Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Precision-Recall plot
    plt.subplot(1, 2, 2)
    for model, color, linestyle in zip(models_data, colors, linestyles):
         # Compute macro-average precision and recall scores, excluding classes with no positive samples
        valid_indices = [i for i in range(model['y_true'].shape[1]) if np.sum(model['y_true'][:, i]) > 0]
        if valid_indices:
            average_precision_macro = np.mean([average_precision_score(model['y_true'][:, i], model['y_score'][:, i]) for i in valid_indices])
        else:
            average_precision_macro = np.nan  # Or handle as needed if no valid classes
        # Compute micro-average precision-recall curve and area
        precision, recall, _ = precision_recall_curve(model['y_true'].ravel(), model['y_score'].ravel())
        average_precision_micro = average_precision_score(model['y_true'], model['y_score'], average="micro")
        average_recall = np.mean(recall)

        plt.plot(recall, precision, color=color, linestyle=linestyle,
             label=f'{model["name"]} (macro AP = {average_precision_macro:.2f}, micro AP = {average_precision_micro:.2f}), AR = {average_recall:.2f})')

    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Summary Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    if (len(file)>0):
        plt.savefig(file, bbox_inches='tight')
    plt.show()

