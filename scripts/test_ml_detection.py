"""
Piarno 打鍵検知 機械学習（純粋NumPy実装）オフライン検証スクリプト
外部DLL依存を排除し、NumPyのみで時系列分類モデル（ロジスティック回帰・2層ニューラルネット）を学習・評価。
"""

import os
import glob
import csv
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "dataset")

def parse_filename(filepath):
    """ファイル名から指名、手、種別を推定"""
    fname = os.path.basename(filepath).lower().replace(" ", "").replace("_", "")
    finger = "unknown"
    for f in ["thumb", "index", "middle", "ring", "pinky"]:
        if f in fname:
            finger = f
            break
            
    hand = "left" if "left" in fname else "right"
    kind = "other"
    if "tap" in fname:
        kind = "tap"
    elif "negslide" in fname:
        kind = "neg_slide"
    elif "negwrist" in fname:
        kind = "neg_wrist"
    elif "negidle" in fname:
        kind = "neg_idle"
    elif "negair" in fname:
        kind = "neg_air"
        
    return finger, hand, kind

def read_csv_data(fpath, kind):
    """CSVを読み込み、NumPy配列の辞書を返す"""
    rows = []
    with open(fpath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        header = [h.strip().lower() for h in header]
        
        req = ["timestamp", "rx", "ry", "rz", "vx", "vy", "vz", "label"]
        col_indices = {}
        for r in req:
            if r in header:
                col_indices[r] = header.index(r)
            else:
                return None
                
        for row in reader:
            if not row or len(row) <= max(col_indices.values()):
                continue
            try:
                vals = [float(row[col_indices[k]]) for k in req]
                rows.append(vals)
            except ValueError:
                continue
                
    if not rows:
        return None
        
    arr = np.array(rows)
    data = {
        "timestamp": arr[:, 0],
        "rx": arr[:, 1],
        "ry": arr[:, 2],
        "rz": arr[:, 3],
        "vx": arr[:, 4],
        "vy": arr[:, 5],
        "vz": arr[:, 6],
        "label": arr[:, 7].astype(int)
    }
    
    if kind != "tap":
        data["label"][:] = 0
        
    return data

def extract_window_features(data, window_size=6, expand_labels=True):
    """
    過去 window_size フレームの特徴量を結合して時系列特徴量を作成
    特徴量: 各フレームの [ry, vy, rz, vz, vx, 加速度ay]
    expand_labels: 打鍵ラベルの前後1コマ（±1フレーム）も打鍵ゾーンとして正解化
    """
    n = len(data["label"])
    if n < window_size:
        return np.empty((0, window_size * 6)), np.empty((0,))
    
    ry = data["ry"]
    vy = data["vy"]
    rz = data["rz"]
    vz = data["vz"]
    vx = data["vx"]
    ay = np.zeros_like(vy)
    ay[1:] = np.diff(vy)
    raw_labels = data["label"]
    
    # 打鍵ラベルの前後1コマを拡張 (打鍵インパクト区間として認識)
    labels = np.copy(raw_labels)
    if expand_labels:
        hit_indices = np.where(raw_labels == 1)[0]
        for idx in hit_indices:
            start = max(0, idx - 1)
            end = min(n, idx + 2)
            labels[start:end] = 1
    
    features = []
    target_labels = []
    
    for i in range(window_size - 1, n):
        w_ry = ry[i - window_size + 1 : i + 1]
        w_vy = vy[i - window_size + 1 : i + 1]
        w_rz = rz[i - window_size + 1 : i + 1]
        w_vz = vz[i - window_size + 1 : i + 1]
        w_vx = vx[i - window_size + 1 : i + 1]
        w_ay = ay[i - window_size + 1 : i + 1]
        
        feat = np.concatenate([w_ry, w_vy, w_rz, w_vz, w_vx, w_ay])
        features.append(feat)
        target_labels.append(labels[i])
        
    return np.array(features), np.array(target_labels)

# ----------------------------------------------------
# 純粋NumPyによる機械学習モデル
# ----------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))

class SimpleLogisticRegression:
    """NumPyによる正則化・重み付きロジスティック回帰"""
    def __init__(self, lr=0.02, epochs=250, l2=0.01, pos_weight=5.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.pos_weight = pos_weight
        self.w = None
        self.b = 0.0
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # 重み（不均衡対策：打鍵サンプルを重視）
        sample_weights = np.where(y == 1, self.pos_weight, 1.0)
        
        for _ in range(self.epochs):
            linear = np.dot(X, self.w) + self.b
            preds = sigmoid(linear)
            error = (preds - y) * sample_weights
            
            grad_w = (np.dot(X.T, error) / n_samples) + self.l2 * self.w
            grad_b = np.sum(error) / n_samples
            
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
    def predict_proba(self, X):
        linear = np.dot(X, self.w) + self.b
        return sigmoid(linear)
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class SimpleMLP:
    """NumPyによる2層ニューラルネットワーク (Input -> Hidden 24 -> Output 1)"""
    def __init__(self, hidden_dim=24, lr=0.01, epochs=300, pos_weight=5.0):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
    def fit(self, X, y):
        np.random.seed(42)
        n_samples, in_dim = X.shape
        # He初期化
        self.W1 = np.random.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = 0.0
        
        sample_weights = np.where(y == 1, self.pos_weight, 1.0).reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        
        for _ in range(self.epochs):
            # Forward
            z1 = np.dot(X, self.W1) + self.b1
            a1 = np.maximum(0, z1) # ReLU
            z2 = np.dot(a1, self.W2) + self.b2
            preds = sigmoid(z2)
            
            # Backward
            delta2 = (preds - y_col) * sample_weights
            grad_W2 = np.dot(a1.T, delta2) / n_samples
            grad_b2 = np.sum(delta2) / n_samples
            
            delta1 = np.dot(delta2, self.W2.T) * (z1 > 0)
            grad_W1 = np.dot(X.T, delta1) / n_samples
            grad_b1 = np.sum(delta1, axis=0) / n_samples
            
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            
    def predict_proba(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return sigmoid(z2).flatten()
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def calc_metrics(y_true, y_pred):
    """適合率・再現率・F1の計算"""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return prec, rec, f1, tp, fp, fn, tn

def evaluate_finger(records, target_finger="index", window_size=6):
    print(f"\n==================================================")
    print(f" 検証対象指: {target_finger.upper()} (時系列ウィンドウ: {window_size}フレーム)")
    print(f"==================================================")
    
    finger_records = [r for r in records if r["finger"] == target_finger]
    if not finger_records:
        print(f"{target_finger} のデータが見つかりません。")
        return
        
    print("データセット一覧:")
    for r in finger_records:
        print(f"  - [{r['kind']:<10}] {r['filename']:<45} ({r['total_frames']}コマ, 打鍵マーク: {r['hit_count']})")
        
    X_list = []
    y_list = []
    file_ids = []
    kinds_list = []
    raw_ry_list = []
    raw_vy_list = []
    
    for f_idx, r in enumerate(finger_records):
        feat, lbl = extract_window_features(r["data"], window_size=window_size)
        if len(lbl) == 0:
            continue
        X_list.append(feat)
        y_list.append(lbl)
        file_ids.extend([f_idx] * len(lbl))
        kinds_list.extend([r["kind"]] * len(lbl))
        
        raw_ry_list.append(r["data"]["ry"][window_size - 1:])
        raw_vy_list.append(r["data"]["vy"][window_size - 1:])
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    file_ids = np.array(file_ids)
    kinds = np.array(kinds_list)
    raw_ry = np.concatenate(raw_ry_list)
    raw_vy = np.concatenate(raw_vy_list)
    
    # 標準化（Z-score正規化：モデル学習用）
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7
    X_norm = (X - mean) / std
    
    total_hits = int(np.sum(y == 1))
    total_non_hits = int(np.sum(y == 0))
    print(f"\n総データ量: {len(y)} コマ (打鍵マーク: {total_hits}コマ, 非打鍵: {total_non_hits}コマ)")
    
    if total_hits < 5:
        print("打鍵データが少ないためスキップします。")
        return

    # ----------------------------------------------------
    # 1. 現行ルールベース (閾値判定) の精度
    # ----------------------------------------------------
    tap_ry_median = np.median(raw_ry[y == 1])
    # 打鍵深さ >= 中央値の75% かつ 速度 >= 1.0
    rule_preds = ((raw_ry >= (tap_ry_median * 0.70)) & (raw_vy >= 1.2)).astype(int)
    
    r_prec, r_rec, r_f1, r_tp, r_fp, r_fn, _ = calc_metrics(y, rule_preds)
    r_fp_slide = int(np.sum((rule_preds == 1) & (y == 0) & (kinds == "neg_slide")))
    r_fp_wrist = int(np.sum((rule_preds == 1) & (y == 0) & (kinds == "neg_wrist")))
    r_fp_air   = int(np.sum((rule_preds == 1) & (y == 0) & (kinds == "neg_air")))
    r_fp_idle  = int(np.sum((rule_preds == 1) & (y == 0) & (kinds == "neg_idle")))
    
    print("\n--------------------------------------------------")
    print("【1. 現行ルールベース (位置ry ＋ 速度vy の閾値)】")
    print(f"  適合率 (Precision) : {r_prec*100:.1f}%  (鳴ったうち本当に打鍵だった割合)")
    print(f"  再現率 (Recall)    : {r_rec*100:.1f}%  (打鍵を逃さず鳴らせた割合)")
    print(f"  F1 スコア          : {r_f1:.3f}")
    print(f"  誤検知内訳: スライド誤爆={r_fp_slide}件, 手首誤爆={r_fp_wrist}件, 空中誤爆={r_fp_air}件, 静止誤爆={r_fp_idle}件")

    # ----------------------------------------------------
    # 2. 未知ファイル交差検証 (Leave-One-File-Out)
    # ----------------------------------------------------
    unique_files = np.unique(file_ids)
    
    for model_name, ModelCls, kwargs in [
        ("ロジスティック回帰 (時系列線形結合)", SimpleLogisticRegression, {"lr": 0.03, "epochs": 200, "pos_weight": 6.0}),
        ("小型ニューラルネット (2層MLP)", SimpleMLP, {"hidden_dim": 24, "lr": 0.03, "epochs": 300, "pos_weight": 8.0})
    ]:
        probs_all = np.zeros(len(y))
        
        # ファイル単位で1つをテスト、残りを学習にする厳格なLeave-One-File-Out評価
        for test_fid in unique_files:
            train_mask = file_ids != test_fid
            test_mask = file_ids == test_fid
            
            if np.sum(y[train_mask] == 1) == 0:
                continue
                
            model = ModelCls(**kwargs)
            model.fit(X_norm[train_mask], y[train_mask])
            probs_all[test_mask] = model.predict_proba(X_norm[test_mask])
            
        # 最適な判定閾値を探索 (0.1 〜 0.7)
        best_th = 0.5
        best_f1 = -1.0
        best_prec, best_rec = 0, 0
        best_preds = None
        
        for th in np.arange(0.15, 0.75, 0.05):
            preds = (probs_all >= th).astype(int)
            p, r, f, _, _, _, _ = calc_metrics(y, preds)
            if f > best_f1:
                best_f1 = f
                best_th = th
                best_prec = p
                best_rec = r
                best_preds = preds
                
        fp_slide = int(np.sum((best_preds == 1) & (y == 0) & (kinds == "neg_slide")))
        fp_wrist = int(np.sum((best_preds == 1) & (y == 0) & (kinds == "neg_wrist")))
        fp_air   = int(np.sum((best_preds == 1) & (y == 0) & (kinds == "neg_air")))
        fp_idle  = int(np.sum((best_preds == 1) & (y == 0) & (kinds == "neg_idle")))
        
        print("\n--------------------------------------------------")
        print(f"【2. 機械学習: {model_name} (最適閾値: {best_th:.2f})】")
        print(f"  適合率 (Precision) : {best_prec*100:.1f}%  (鳴ったうち本当に打鍵だった割合)")
        print(f"  再現率 (Recall)    : {best_rec*100:.1f}%  (打鍵を逃さず鳴らせた割合)")
        print(f"  F1 スコア          : {best_f1:.3f}")
        print(f"  誤検知内訳: スライド誤爆={fp_slide}件, 手首誤爆={fp_wrist}件, 空中誤爆={fp_air}件, 静止誤爆={fp_idle}件")

def main():
    csv_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.csv*"), recursive=True)
    records = []
    for fpath in csv_files:
        finger, hand, kind = parse_filename(fpath)
        data = read_csv_data(fpath, kind)
        if data is None:
            continue
        records.append({
            "filepath": fpath,
            "filename": os.path.basename(fpath),
            "finger": finger,
            "hand": hand,
            "kind": kind,
            "data": data,
            "hit_count": int(np.sum(data["label"] == 1)),
            "total_frames": len(data["label"])
        })
        
    evaluate_finger(records, target_finger="index", window_size=6)
    evaluate_finger(records, target_finger="middle", window_size=6)

if __name__ == "__main__":
    main()
