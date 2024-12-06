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
    qs = np.array([-1 * np.log(ws[j])*(n - (j+1) - (p + q + 1)/2) for j in range(m)])  # 统计量
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