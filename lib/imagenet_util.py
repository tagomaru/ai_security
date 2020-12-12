import numpy as np
from tensorflow.keras.applications.resnet50 import decode_predictions

def get_top_pred(Y_hat):
    """
    ImageNet のモデルの推論結果から、上位一位のクラスのインデックス、名前、スコアを取得し、返す
    Args:
        Y_hat (ndarray): ImageNet　のモデルのデコード前の推論結果 `バッチサイズ, 1000`
    Returns:
        index (int): クラスのインデックス
        name (str): クラスの名前
        score (float): スコア
    """
    decoded_Y_hat = decode_predictions(Y_hat, top=1)
    index = np.argmax(Y_hat, axis=1)[0]
    name = decoded_Y_hat[0][0][1]
    score = decoded_Y_hat[0][0][2]
    return index, name, score

def postprocess_imagenet(X):
    """
    ImageNet 用のポストプロセス
    Args:
        X (ndarray): ImageNet 用に正規化されたデータ。バッチと非バッチそれぞれに対応
    Returns:
        denormalized_X (ndarray): 非正規化データ
    """
    # ImageNet の BGR の平均値
    bgr_mean_imagenet = np.array([103.939, 116.779, 123.68], dtype='float32')
    
    # 非正規化
    denormalized_X = (X + bgr_mean_imagenet)[..., ::-1]
    
    # クリッピング
    denormalized_X = np.clip(denormalized_X, 0, 255)
    
    return denormalized_X

def clip_imagenet(X):
    """
    ImageNet 用に正規化されたデータのクリッピングを行う。
    Args:
        X (ndarray): ImageNet 用に正規化されたデータ。バッチと非バッチそれぞれに対応
    Returns:
        clippped_X (ndarray): クリッピングされた正規化データ
    """
    # ImageNet の Mean BGR (≠ RGB)
    bgr_mean_imagenet = np.array([103.939, 116.779, 123.68], dtype='float32')
    
    # クリッピング
    clippped_X = np.clip(X, 0 - bgr_mean_imagenet, 255.0 - bgr_mean_imagenet)
    
    return clippped_X