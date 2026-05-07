# from . import util
from .util import parse_wrd_file, get_sinusoidal_positional_embeddings, save_embeddings, get_mel_feats
import os
import numpy as np


def extract_sinusoidal_embeddings(file_names, input_dir, output_file, d_model=50):
    """
    Extract sinusoidal positional embeddings for multiple files and save result.
    Args:
        file_names: List of file names (str, without extension).
        input_dir: Directory containing .wrd files.
        output_file: Path to save the final concatenated embeddings.
        d_model: Dimensionality of embeddings.
    Returns:
        final_embeddings: Numpy array of shape (N, d), where N=total words and d=d_model.
    """
    all_embeddings = []

    for file_name in file_names:
        wrd_file = os.path.join(input_dir, file_name + ".wrd")

        # Parse words and timestamps
        words, _ = parse_wrd_file(wrd_file)

        # Compute sinusoidal positional embeddings for the words
        embeddings = get_sinusoidal_positional_embeddings(words, d_model)
        all_embeddings.append(embeddings)

    # Concatenate all embeddings into a single NxD matrix
    final_embeddings = np.vstack(all_embeddings)

    # Save the embeddings
    save_embeddings(final_embeddings, output_file)

    return final_embeddings


def extract_word_level_mel_features(file_names, input_dir, output_file, n_fft=400, hop_length=160, n_mels=128):
    """
    Extract word-level Mel-spectrogram features for multiple files and save result.

    Args:
        file_names (list of str): List of file names (without extension).
        input_dir (str): Directory containing the .wrd and .wav files.
        output_file (str): Path to save the final concatenated word-level Mel features.
        n_fft (int, optional): FFT window size (in samples, default=400 for 25ms at 16 kHz).
        hop_length (int, optional): Number of samples between successive frames (default=160 for 10ms at 16 kHz).
        n_mels (int, optional): Number of Mel bands to generate (default=128).

    Returns:
        final_word_features (np.ndarray): Array of shape (N, n_mels) where N=total words in all files.
    """

    # Initialize a list to store features for all files
    all_word_features = []

    for file_name in file_names:
        # Paths for the .wrd and .wav files
        wrd_file = os.path.join(input_dir, file_name + ".wrd")
        wav_file = os.path.join(input_dir, file_name + ".wav")
        
        # Parse words and timestamps from the .wrd file
        start_end_list = []  # List to store (start, end) in samples
        with open(wrd_file, "r") as f:
            for line in f:
                fields = line.strip().split()
                start = int(fields[0])  # Start time in samples
                end = int(fields[1])    # End time in samples
                start_end_list.append((start, end))

        # Compute Mel-spectrogram features using the helper function
        _, word_features = get_mel_feats(wav_file, start_end_list, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Append the word features to the list
        all_word_features.extend(word_features)

    # Stack all word features into a single array
    final_word_features = np.vstack(all_word_features)

    # Save the word-level Mel features
    save_embeddings(final_word_features, output_file)

    return final_word_features


def extract_glove_features_from_wrd(timit_names, timit_dir, word_index, embedding_matrix):
    """
    Extract GloVe word embeddings from .wrd files.

    Args:
        timit_names (list of list of str): List of file blocks containing file names without extensions.
        timit_dir (str): Directory where .wrd files are located.
        word_index (dict): A dictionary mapping words to their indices in the embedding matrix.
        embedding_matrix (np.ndarray): Pretrained embedding matrix (e.g., GloVe embeddings).

    Returns:
        dict: A dictionary containing:
            - "feat": A concatenated NumPy array of word embeddings (shape: [N, embedding_dim]).
            - "word_seq_list": A list of word identifiers with unique indices (e.g., ["word_0", "word_1"]).
    """
    non_padding_hd_list = []  # List to store word-level embeddings
    word_seq_list = []  # List to store unique word identifiers
    word_count_dict = {}  # Used to ensure unique identifiers for repeated words

    for i, timit_block in enumerate(timit_names):
        print(f"Processing block {i + 1}/{len(timit_names)}...")
        for j, file_name in enumerate(timit_block):
            print(f"  Processing file: {file_name}")

            # Parse the .wrd file
            wrd_file_path = os.path.join(timit_dir, file_name + ".wrd")
            words, _ = parse_wrd_file(wrd_file_path)  # Extract words only (timestamps are not needed for GloVe)

            # Process each word in the .wrd file
            non_padding_hd = []
            for word in words:
                word = word.lower()  # Convert to lowercase for consistency
                if word in word_index:
                    # Get the embedding vector from the pre-trained embedding matrix
                    embedding_vector = embedding_matrix[word_index[word]]
                    non_padding_hd.append(embedding_vector)

                    # Generate a unique identifier for the word
                    word_count_dict.setdefault(word, 0)
                    word_seq_list.append(f"{word}_{word_count_dict[word]}")
                    word_count_dict[word] += 1
                else:
                    print(f"  Warning: Word '{word}' not found in embedding vocabulary.")

            if non_padding_hd:
                non_padding_hd_list.append(np.stack(non_padding_hd))

    # Concatenate all embeddings into a single NumPy array
    feat = np.concatenate(non_padding_hd_list, axis=0) if non_padding_hd_list else np.array([])

    return {"feat": feat, "word_seq_list": word_seq_list}


def extract_wav2vec2_features(model, speech_array, sample_rate=16000):
    """
    Extract Wav2Vec2 features from audio data. Includes features from the 
    feature extractor (CNN layers), projection layer, and encoder hidden states.

    Args:
        model: Pretrained Wav2Vec2 model.
        speech_array (torch.Tensor): Tensor containing the audio signal (batch size = 1).
        sample_rate (int): Sampling rate of the input audio.

    Returns:
        dict: A dictionary containing extracted features:
            - 'ext': Output from the feature extractor (last CNN layer).
            - 'proj': Output from the projection layer.
            - 'encoder': List of hidden states from all encoder layers.
            - 'pos_embs': Positional convolution embeddings from the encoder.
    """
    # Placeholder to store intermediate outputs
    lat = []
    lat.append(speech_array.unsqueeze_(0))  # Add batch dimension for Wav2Vec2 processing
    
    # Process through the feature extractor (CNN layers)
    for conv_layer in model.wav2vec2.feature_extractor.conv_layers:
        lat.append(conv_layer(lat[-1]))

    # Obtain feature projection outputs
    lat_p = model.wav2vec2.feature_projection(lat[-1].permute(0, 2, 1))

    # Get positional convolution embeddings
    pos_conv_emb = model.wav2vec2.encoder.pos_conv_embed(lat_p[0])

    # Get all encoder layer hidden states
    lat_e = model.wav2vec2.encoder(lat_p[0], output_hidden_states=True)

    # Package all features into a dictionary
    features = {
        "ext": np.squeeze(lat[-1].cpu().detach().numpy()).T,         # Last CNN layer outputs
        "proj": np.squeeze(lat_p[0].cpu().detach().numpy()),        # Projection outputs
        "encoder": [np.squeeze(h.cpu().detach().numpy()) for h in lat_e.hidden_states],  # Hidden states
        "pos_embs": np.squeeze(pos_conv_emb.cpu().detach().numpy())  # Positional embeddings
    }

    return features
