import os

from functools import wraps
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy.stats import zscore
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sklearn.model_selection as model_selection
# import svcca.pwcca as pwcca
from scipy.stats import chi2
import pickle
import joblib
import re
import soundfile as sf
import torch 

def load_glove_embeddings(filepath, word_index, embedding_dim):
    """
    Load GloVe embeddings and create a matrix for the given vocabulary.

    Args:
        filepath (str): Path to the GloVe embedding file.
        word_index (dict): A dictionary mapping words to their indices.
        embedding_dim (int): The dimension of the GloVe embeddings.

    Returns:
        np.ndarray: An embedding matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(word_index) + 1  # +1 for reserved index 0
    embedding_matrix = np.zeros((vocab_size, embedding_dim))  # Initialize matrix with zeros

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def get_mel_feats(audio_path, start_end_list, hop_length=160, n_fft=400, n_mels=128):
    """
    Extract Mel-spectrogram features for an audio file and compute word-level features
    based on start and end positions in the audio (given in sample indices).

    Args:
        audio_path (str): Path to the audio file.
        start_end_list (list of tuples): A list of (start_sample, end_sample) tuples.
            Each tuple represents the start and end sample indices of words in the audio.
        hop_length (int, optional): Number of audio samples between successive Mel-spectrogram frames.
            Default is 160, corresponding to 10ms at a 16kHz sample rate.
        n_fft (int, optional): Length of the FFT window (in samples). Default is 400 (25ms at 16kHz).
        n_mels (int, optional): Number of Mel bands to generate. Default is 128.

    Returns:
        tuple:
            - sentence_feat (np.ndarray): Full Mel-spectrogram of the audio file with shape (n_mels, time_frames).
            - word_feats (list of np.ndarray): A list where each element is the mean Mel-spectrogram feature vector
              for a specific word, computed over its (start_sample, end_sample) range.

    Notes:
        - This function computes the Mel-spectrogram for the entire audio file first.
        - It normalizes the start and end positions (given in sample indices) to the corresponding frame indices
          in the Mel-spectrogram, then calculates the mean feature representation for each word.

    Example Usage:
        audio_path = "audio/example.wav"
        start_end_list = [(0, 4800), (4800, 8000)]  # Word sample index ranges
        sentence_feat, word_feats = get_mel_feats(audio_path, start_end_list)
    """

    # Read the audio file and obtain the sampling rate
    audio, sr = sf.read(audio_path)
    duration = len(audio)  # Total number of audio samples in the file

    # Compute the Mel-spectrogram using librosa
    # - y: audio time series
    # - sr: sampling rate of the audio
    # - n_fft: FFT window size
    # - hop_length: number of audio samples between successive frames
    # - n_mels: number of Mel frequency bands
    sentence_feat = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Get the number of frames in the resulting Mel-spectrogram
    feat_time_len = sentence_feat.shape[1]

    # Initialize the list to store word-level features
    word_feats = []

    # Iterate over each (start_sample, end_sample) pair in start_end_list
    for s, e in start_end_list:
        # Convert sample indices to frame indices in the Mel-spectrogram
        word_start = round(feat_time_len * (s / duration))  # Start frame index
        word_end = round(feat_time_len * (e / duration))    # End frame index

        # Calculate the mean Mel-spectrogram feature across the word's frames
        word_feats.append(np.mean(sentence_feat[:, word_start:word_end], axis=1))

    # Return the full Mel-spectrogram and word-level features
    return sentence_feat, word_feats


def get_feature_names(model_name):
    """_summary_

    Args:
        model_name (_type_): _description_

    Returns:
        list: _description_
    """
    model_name_list = ["wav2vec2", "wav2vec2_xlsr", "wav2vec2_xlsr_en", "hubert", "gpt", "glove", "glove_oov", "mel", "gpt2xl", "residual_context",
                       "wav2vec2_base_lang_id", "wav2vec2_xlsr_large_gender_recognition", "wav2vec2_xlsr_large_emotion_recognition", "w2v_large_robust_en", "w2v_large_cn"] +\
                        [f"wav2vec2_xlsr_ft_{i}" for i in range(1, 11)]  + [f"wav2vec2_xlsr_librispeech_pretrain_{i}" for i in range(1, 11)] +\
                        ["gpt2xl_ccs", "wav2vec2_ccs", "gpt2xl_rest_ccs", "wav2vec2_rest_ccs"] + ["gpt2xlCCs_glove_ccs", "gpt2xlCCs_residual_context_ccs", "gpt2xlCCs_mel_ccs"]
    assert model_name in model_name_list

    if model_name in ["wav2vec2", "wav2vec2_base_lang_id"]:
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5',
                       'encoder6', 'encoder7', 'encoder8', 'encoder9', 'encoder10', 'encoder11', 'encoder12']
    elif model_name in ["wav2vec2_xlsr", "wav2vec2_xlsr_large_gender_recognition", "wav2vec2_xlsr_large_emotion_recognition", "w2v_large_robust_en", "w2v_large_cn"]:
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder2', 'encoder4', 'encoder6', 'encoder8', 'encoder10',
                       'encoder12', 'encoder14', 'encoder16', 'encoder18', 'encoder20', 'encoder22', 'encoder24']
    elif model_name in [f"wav2vec2_xlsr_ft_{i}" for i in range(1, 11)]:
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder2', 'encoder4', 'encoder6', 'encoder8', 'encoder10',
                       'encoder12', 'encoder14', 'encoder16', 'encoder18', 'encoder20', 'encoder22', 'encoder24']
    elif model_name in [f"wav2vec2_xlsr_librispeech_pretrain_{i}" for i in range(1, 11)]:
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder2', 'encoder4', 'encoder6', 'encoder8', 'encoder10',
                       'encoder12', 'encoder14', 'encoder16', 'encoder18', 'encoder20', 'encoder22', 'encoder24']
    elif model_name == "wav2vec2_xlsr_en":
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder2', 'encoder4', 'encoder6', 'encoder8', 'encoder10',
                       'encoder12', 'encoder14', 'encoder16', 'encoder18', 'encoder20', 'encoder22', 'encoder24']
    elif model_name == "hubert":
        nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5',
                       'encoder6', 'encoder7', 'encoder8', 'encoder9', 'encoder10', 'encoder11']
    elif model_name == "gpt":
        nn_features = ['fs_ext', 'decoder0', 'decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5',
                       'decoder6', 'decoder7', 'decoder8', 'decoder9', 'decoder10', 'decoder11']
        #nn_features = ["decoder0"]
    elif model_name in ["gpt2xl_ccs", "wav2vec2_ccs", "gpt2xl_rest_ccs", "wav2vec2_rest_ccs"]:  
        nn_features = ["decoder8_encoder7"]
    elif model_name in ["glove", "glove_oov", "mel", "residual_context", "gpt2xlCCs_glove_ccs", "gpt2xlCCs_residual_context_ccs", "gpt2xlCCs_mel_ccs"]:
        nn_features = ['feat']
    elif model_name == "gpt2xl":
        nn_features = ['fs_ext', 'decoder0', 'decoder4', 'decoder8', 'decoder12', 'decoder16', 'decoder20', 
                 'decoder24', 'decoder28', 'decoder32', 'decoder36', 'decoder40', 'decoder44', 'decoder47']
        
    return nn_features


def scale_and_pca_one_dataset(feat, variance_ratio=0.99, pca_dim=None, scale=True):
    scaler = StandardScaler(with_std=scale)  # normalize
    trainning_set = scaler.fit_transform(feat)
    if pca_dim:
        pca = PCA(pca_dim)  # applying pca
    else:
        pca = PCA(variance_ratio)  # applying pca
    
    rs = pca.fit_transform(trainning_set)
    return rs


def cca_corr_sig_test(cca_corrs, n, p, q, n_comps=None, alpha=0.05):
    """
    Args:
        corrs (_type_): canonical correlation on test set, length = min(p, q)
        p (_type_): feat dim of X
        q (_type_): feat dim of Y
        n (_type_): num of samples
        n_comps (_type_): number of canonical components, if None, set as min(p, q)
        alpha (_type_): significance level
    """
    m = n_comps if n_comps else min(p, q)
    
    lambdas = 1/(1 - np.array(cca_corrs)**2) - 1
    ws = np.zeros((m,))
    ws[-1] = 1/(1+lambdas[-1])
    for i in range(m-2, -1, -1):
        ws[i] = 1/(1+lambdas[i])*ws[i+1]
    # print(ws)
    dfs = [(p - j + 1)*(q - j + 1) for j in range(1, m+1)]
    
    chis = np.array([chi2.ppf(1 - alpha, dfs[j]) for j in range(len(dfs))])
    qs = np.array([-1 * np.log(ws[j])*(n - (j+1) - (p + q + 1)/2) for j in range(m)]) 
    tmp = qs - chis
    is_sig = tmp > 0
    return chis, qs, is_sig


def save(path, name, *results_keys):
    def decorator(some_function):
        @wraps(some_function)
        def wrapper(*args, **kwargs):
            results = some_function(*args, **kwargs)
            if type(results) is np.ndarray:
                results_dict = {results_keys[0]: results}
            else:
                results_dict = dict(zip(results_keys, results))
            subject = kwargs.pop("subject")
            block = kwargs.pop("block", None)
            subject_block = subject + "_B" + str(block)

            metadata = subject_block
            hz = kwargs.pop("hz", None)
            if hz is not None:
                metadata = metadata + "_" + str(hz) + "hz"

            full_path = os.path.join(path, metadata + "_" + name)
            sio.savemat(full_path, results_dict)
            return results
        return wrapper
    return decorator

def get_subject_block(subject, block):
    return subject + "_B" + str(block)

def get_mel_spectrogram_for_wavpath(wavpath, time_bin=10, n_mels=128):
    fs, y = wavfile.read(wavpath)
    if len(y.shape) > 1 and y.shape[1] == 2:
        y = y[:, 0]
    assert fs/100 == 160
    hz = 1000/time_bin
    assert hz == int(hz)
    hop_length = fs/hz
    assert hop_length == int(hop_length)
    S = librosa.feature.melspectrogram(y.astype(np.float), fs, fmax=8000, hop_length=int(hop_length), n_mels=n_mels)
    S = zscore(librosa.power_to_db(S), axis=1)
    return S

def get_mels(n_mels=128, fmin=0, fmax=8000, round=True):
    """Returns center frequencies of mel bands in kHz
    """
    if round:
        return np.around(librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax), -2)/1000
    else:
        return librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)/1000

def time_to_index(t, hz=100):
    return np.round(t * hz).astype(np.int)

def index_to_time(i, hz=100):
    return i / hz


def scale_and_pca_save_and_load(dstim, train, test, valid_len, variance_ratio=0.95, pca_dim=50, apply_pca_dim=False, 
                                scale=True, subject="", fold_idx="", feat_name="", model_name="", scaler_path="", pca_path=""):
    """save scaler and pca model if path not exists, else load

    Args:
        dstim (_type_): (n_samples, n_features)
        train (_type_): index of trainning set in dstim
        test (_type_): index of test set in dstim
        valid_len (_type_): _description_
        variance_ratio (float, optional): _description_. Defaults to 0.95.
        pca_dim (int, optional): _description_. Defaults to 50.
        apply_pca_dim (bool, optional): False - use variance_ratio, True - use pca_dim
        scale (bool, optional): True - scale, False - center
        subject (str, optional): _description_. Defaults to "".
        fold_idx (str, optional): _description_. Defaults to "".
        path (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    
    trainning_set = dstim[train[:len(train)-valid_len], :]
    test_set = dstim[test, :]
    validation_set = dstim[train[-1*valid_len:], :]

    print("\n pca train shape: {}, test shape: {}, valid shape: {}".format(trainning_set.shape, test_set.shape, validation_set.shape))
    s = time.time()
    scaler_path = scaler_path.replace("subject", subject).replace("fold_idx", fold_idx).replace("feat_name", feat_name).replace("model_name", model_name)
    pca_path = pca_path.replace("subject", subject).replace("fold_idx", fold_idx).replace("feat_name", feat_name).replace("model_name", model_name)
    print(scaler_path, pca_path)
    if os.path.exists(scaler_path):
        print("scaler model exists : ", scaler_path)
        scaler = joblib.load(scaler_path) 
        pca = joblib.load(pca_path) 
        
        trainning_set = scaler.transform(trainning_set)
        test_set = scaler.transform(test_set)
        validation_set = scaler.transform(validation_set)
        
        trainning_set = pca.transform(trainning_set)
        test_set = pca.transform(test_set)
        validation_set = pca.transform(validation_set)
    else:
        scaler = StandardScaler(with_std=scale)  # normalize
        trainning_set = scaler.fit_transform(trainning_set)
        test_set = scaler.transform(test_set)
        validation_set = scaler.transform(validation_set)

        
        # with open("/root/neuro/results/trainningset_tmp.pkl", "wb") as f:
        #     pickle.dump(trainning_set, f)
        if not apply_pca_dim:
            pca = PCA(variance_ratio)  # applying pca
        else:
            print("apply pca_dim")
            pca = PCA(n_components=pca_dim)
        trainning_set = pca.fit_transform(trainning_set)
        test_set = pca.transform(test_set)
        validation_set = pca.transform(validation_set)
        joblib.dump(scaler, scaler_path) 
        joblib.dump(pca, pca_path) 
        print("save in ", scaler_path)
        
    e = time.time()

    print("dstim PCA: {} components: {}".format(variance_ratio, pca.n_components_))
    print("scale_and_pca consume time: {} s".format(e - s))
    return trainning_set, test_set, validation_set


def scale_and_pca(dstim, train, test, valid_len, variance_ratio=0.95, pca_dim=50, apply_pca_dim=False, scale=True):
    """
        input:
            dstim: (n_samples, n_features)
            train: index of trainning set in dstim
            test: index of test set in dstim
            apply_pca_dim: False - use variance_ratio, True - use pca_dim
            scale: True - scale, False - center
    """
    trainning_set = dstim[train[:len(train)-valid_len], :]
    test_set = dstim[test, :]
    validation_set = dstim[train[-1*valid_len:], :]

    print("\n pca train shape: {}, test shape: {}, valid shape: {}".format(trainning_set.shape, test_set.shape, validation_set.shape))
    s = time.time()
    scaler = StandardScaler(with_std=scale)  # normalize
    trainning_set = scaler.fit_transform(trainning_set)
    test_set = scaler.transform(test_set)
    validation_set = scaler.transform(validation_set)

    if not apply_pca_dim:
        pca = PCA(variance_ratio)  # applying pca
    else:
        print("apply pca_dim")
        pca = PCA(n_components=pca_dim)
    trainning_set = pca.fit_transform(trainning_set)
    test_set = pca.transform(test_set)
    validation_set = pca.transform(validation_set)
    e = time.time()

    print("dstim PCA: {} components: {}".format(variance_ratio, pca.n_components_))
    print("scale_and_pca consume time: {} s".format(e - s))
    return trainning_set, test_set, validation_set


def parse_wrd_file(file_path):
    """
    Parse a .wrd file to extract words and their start/end timestamps.
    Args:
        file_path: Path to the .wrd file.
    Returns:
        A tuple of (words, start_end_list), where:
            - words: List of words in the file.
            - start_end_list: List of (start, end) timestamp tuples.
    """
    words = []
    start_end_list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, "r") as f:
        for line in f:
            match = re.findall(r"(\d+)\s+(\d+)\s+(.*)", line)
            if match:
                start, end, word = int(match[0][0]), int(match[0][1]), match[0][2].strip()
                start_end_list.append((start, end))
                words.append(word)

    return words, start_end_list
        
        
def get_sinusoidal_positional_embeddings(sentence_tokens, d_model=50):
    """
    Compute sinusoidal positional embeddings for a sequence of tokens.
    Args:
        sentence_tokens: List of tokens (words in the sequence).
        d_model: Embedding dimension size.
    Returns:
        A numpy array of shape (len(sentence_tokens), d_model).
    """
    seq_len = len(sentence_tokens)
    pos_embeddings = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_embeddings[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
        for i in range(1, d_model, 2):
            pos_embeddings[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    
    return pos_embeddings


def save_embeddings(embeddings, output_path):
    """
    Save embeddings to a .pkl file.
    """
    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved at {output_path}")
    
    
def get_merged_hidden_states(tokens_from_id, true_tokens, hidden_states, mode="BBPE"):
    """
    Get merged hidden state vectors.
    
    Because the tokenizer of the pre-trained model (like GPT2) differs from the dataset's,
    we need to merge `tokens_from_id` into `true_tokens`. If adjacent tokens are merged, 
    their corresponding hidden states are summed together.

    For now, the BPE (Byte Pair Encoding) or BBPE-style tokenization is used, but the merging step might differ for other tokenization strategies.

    Args:
        tokens_from_id: list of tokens obtained by converting token IDs into tokens 
                        (e.g., ["word</w>", "..."]). 
                        For GPT2, these tokens often use BBPE encoding.
        true_tokens: list of words extracted from the `.wrd` file (ground-truth tokens).
        hidden_states: list of hidden states for each layer of the model. 
                       Each element is a tensor with shape [1, WORD_NUM, HIDDEN_SIZE],
                       where WORD_NUM is the number of tokens and HIDDEN_SIZE is the size of the hidden state vector.
        mode: str, the mode of tokenizer being used ("BBPE" or "BPE"). 
              The merging rules depend on the tokenization style.

    Returns:
        merged_hidden_states: list of merged hidden states for each layer.
                              The shape for each layer is [WORD_NUM_TRUE, HIDDEN_SIZE], 
                              where WORD_NUM_TRUE corresponds to the number of `true_tokens`.
    """
    merged_hidden_states = []  # Final merged hidden states for all layers
    cur_subword_num = 0  # Tracks the number of subwords that form a single true token

    for k in range(len(hidden_states)):  # Loop over all layers of hidden states
        cur_word = ""  # Temporarily stores the reconstructed word
        i, j = 0, 0  # i: index for tokens_from_id, j: index for true_tokens
        target = true_tokens[j].lower()  # The current target token to match with
        v = torch.zeros(1, len(true_tokens), hidden_states[k].shape[-1])  # Hidden states after merging for this layer
        
        while i < len(tokens_from_id) and j < len(true_tokens):  # Iterate through both token lists
            if mode == "BBPE":  # For models using BBPE tokenizer (like GPT2)
                if tokens_from_id[i].startswith("Ġ"):  # Ġ indicates the start of a word in BBPE
                    t = re.findall(r"Ġ(.*)", tokens_from_id[i])[0]  # Remove "Ġ" to get the token
                else:
                    t = tokens_from_id[i]  # No "Ġ", take the token as-is
            elif mode == "BPE":  # For models using standard BPE tokenizer
                if tokens_from_id[i].endswith("</w>"):  # </w> marks the end of a word in BPE
                    t = re.findall(r"(.*)</w>", tokens_from_id[i])[0]
                else:
                    t = tokens_from_id[i]  # No </w>, take the token as-is

            # Skip punctuation except for an apostrophe
            if len(t) == 1 and (t != "'" and not t.isalpha()):
                i += 1
                continue

            # Accumulate the subword into the current reconstructed word
            cur_word += t
            cur_subword_num += 1

            # Add the hidden state for the current token to the merged vector
            v[0, j, :] += hidden_states[k][0, i, :]

            if cur_word == target:  # If the reconstructed word matches the true token
                cur_word = ""  # Reset the reconstructed word
                v[0, j, :] /= cur_subword_num  # Mean-pooling over subword hidden states
                cur_subword_num = 0
                j += 1  # Move to the next true token
                if j < len(true_tokens):  # Update the next target token if available
                    target = true_tokens[j].lower()
            i += 1  # Move to the next token in tokens_from_id

        # Ensure that all tokens have been processed and successfully matched
        assert i == len(tokens_from_id) and j == len(true_tokens), \
            "Mismatch between token indices: not all tokens were successfully merged."

        # Save the merged hidden states for the current layer
        merged_hidden_states.append(v.detach().numpy().squeeze())

    return merged_hidden_states


def compute_compression_ratio(original_len, compressed_len):
    """
    Compute the compression ratio between the original audio signal length and
    the compressed feature length.

    Args:
        original_len (int): Original audio signal length (in samples).
        compressed_len (int): Compressed feature length (in frames).

    Returns:
        float: Compression ratio (compressed_len / original_len).
    """
    return compressed_len / original_len


def segment_features(features, start_end_list, compression_ratio):
    """
    Segment Wav2Vec2 features based on word-level timestamps.

    Args:
        features (dict): Dictionary of features ('ext', 'proj', 'encoder', 'pos_embs').
        start_end_list (list of tuples): List of (start_time, end_time) pairs in seconds (timestamps).
        compression_ratio (float): Ratio of compressed feature length to original audio length.

    Returns:
        dict: Segmented features for each feature type:
            - 'ext': Word embeddings from the feature extractor output.
            - 'proj': Word embeddings from the projection layer output.
            - 'encoder': Word embeddings from all encoder layers.
    """
    # Convert timestamps to features frame indices
    start_end_list_new = [(round(start * compression_ratio), round(end * compression_ratio)) for start, end in start_end_list]

    # Initialize storage
    segmented_features = {"ext": [], "proj": [], "encoder": [[] for _ in features["encoder"]]}
    
    # Segment features for each word
    for token_start, token_end in start_end_list_new:
        if token_start == token_end:  # Skip empty segments
            continue
        
        # Extract and average features for each segment
        segmented_features["ext"].append(features["ext"][token_start:token_end].mean(axis=0))
        segmented_features["proj"].append(features["proj"][token_start:token_end].mean(axis=0))
        
        # Encoder features (one per layer)
        for layer_idx, layer_features in enumerate(features["encoder"]):
            segmented_features["encoder"][layer_idx].append(layer_features[token_start:token_end].mean(axis=0))
    
    return segmented_features