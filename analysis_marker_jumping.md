# マーカー飛び（ランドマーク座標ジャンプ）の原因解析と対策レポート

本ドキュメントは、Piarno（Webカメラ＋MediaPipe Hand Landmarkerによる打鍵検出システム）において、演奏中・待機中に発生する「マーカー飛び（指先や骨格の突発的ジャンプ・揺れ）」のコード解析結果と、その根本原因および改善策をまとめたものです。

---

## 1. エグゼクティブサマリー

解析の結果、マーカー飛びは単一の原因ではなく、**「MediaPipeの追従特性」「異常値ガードの監視漏れ」「適応型フィルター（One Euro Filter）のスパイク追従特性」** の3層が組み合わさることで発生していることが判明しました。

特に最大の要因は、**異常値ジャンプ判定ガードが「手首（Landmark 0）」のみを監視しており、最もブレやすい「指先（TIP）」の突発的ワープを素通りさせている点**です。

---

## 2. 根本原因の詳細解析（コード該当箇所とメカニズム）

### 原因 ①：異常ジャンプ判定ガードが「手首（Landmark 0）」のみしか監視していない【最大要因】

- **該当コード**: `main.js` 1723〜1736行目
```javascript
// 1フレームでの極端な座標ジャンプ（Palm Detection誤検出や左右反転ワープ）の遮断ガード
let isAnomalousJump = false;
if (hasHands && lastRawLandmarks) {
  const rawWrist = landmarks[0];
  const prevWrist = lastRawLandmarks[0];
  const jumpDist = Math.hypot((rawWrist.x - prevWrist.x) * width, (rawWrist.y - prevWrist.y) * height);
  // 1フレーム（約16ms）で画面幅の20%以上手首がワープした場合はAIの誤検出フレームとして直前座標を維持
  if (jumpDist > width * 0.20) {
    isAnomalousJump = true;
    landmarks = lastRawLandmarks;
  }
}
```

- **メカニズムと問題点**:
  1. プレイヤーの手首（Wrist）は机面付近に安定して置かれているため、`jumpDist > width * 0.20`（画面幅の20%以上の移動）を超えることは滅多にありません。
  2. しかし、水平アングル（机面スレスレ）では、**打鍵時や指の屈伸時に「指先（TIP: 4, 8, 12, 16, 20）」が机面の影や隣の指と重なり合い、AIが指先の前後・左右関係を1フレームだけ誤認して数十〜数百ピクセル跳ね上がる現象**が頻発します。
  3. 手首が動いていないため、この指先の突発的なワープは異常値ガードを完全にすり抜け、そのまま平滑化フィルターに入力されてしまいます。

---

### 原因 ②：1 Euro Filter の「高速追従特性」による外れ値増幅

- **該当コード**: `main.js` 274〜322行目 (`OneEuroFilter.filter`)
```javascript
// 速度（1階差分）の平滑化
const dx = (x - this.xPrev) / dt;
const aD = this.alpha(this.dCutoff, dt);
const dxHat = aD * dx + (1 - aD) * this.dxPrev;
this.dxPrev = dxHat;

// 速度に応じた動的カットオフ周波数（素早い動きでは自動で高くなり遅延ゼロ）
const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
const a = this.alpha(cutoff, dt);

// 信号の平滑化
const xHat = a * x + (1 - a) * this.xPrev;
```

- **メカニズムと問題点**:
  1. 1 Euro Filter は「指が止まっている時は強く平滑化し、素早く動いた時は遅延ゼロで追従する」設計です。
  2. AIが1フレームだけ誤った外れ値（スパイク）を出力すると、速度微分 `dx = (x - this.xPrev) / dt` が瞬間的に極大化します。
  3. フィルターはこれを「人間の指が超高速で動いた」と誤認し、動的カットオフ周波数 `cutoff` を一気に引き上げて平滑化を解除し、生の外れ値にそのまま追従してしまいます。
  4. 人間の指の物理限界速度を超えるような移動量に対する**「最大移動速度リミッター（クランプ）」が存在しない**ため、マーカーの跳ね上がりを抑え込めません。

---

### 原因 ③：検出ロスト（1〜3フレーム補間）復帰時のタイムスタンプ差と速度跳ね上がり

- **該当コード**: `main.js` 1689〜1712行目、1739〜1745行目
```javascript
if (!hasHands) {
  lostFrames++;
  if (lostFrames <= MAX_LOST_FRAMES && hasValidSmoothedLandmarks) {
    // 直前フレームの座標をそのまま使用して継続描画
  } else { ... }
}
```

- **メカニズムと問題点**:
  1. 手の検出が1〜3フレーム途切れた際、描画は直前座標（`smoothedLandmarksPool`）を維持してちらつきを防いでいます。
  2. しかし、このロスト期間中は `landmarkFilters[i].filter()` の更新がスキップされるため、フィルター内部の前回タイムスタンプ `tPrev` が停止します。
  3. 2〜3フレーム後に手が再検出された瞬間、経過時間 `dt = (now - tPrev)` が通常の2〜3倍（約33〜50ms）に拡大した状態で新座標が入力されます。
  4. その結果、フィルターの内部速度 `dxHat` の算出に不連続な段差が生じ、復帰初フレームでマーカーが一瞬ピクッと跳ねる原因になります。

---

### 原因 ④：MediaPipe HandLandmarker の追従・存在信頼度閾値が低め

- **該当コード**: `main.js` 1360〜1370行目
```javascript
handLandmarker = await HandLandmarker.createFromOptions(vision, {
  runningMode: "VIDEO",
  numHands: 1,
  minHandDetectionConfidence: 0.4,
  minHandPresenceConfidence: 0.4,
  minTrackingConfidence: 0.3
});
```

- **メカニズムと問題点**:
  1. `minTrackingConfidence: 0.3` および `minHandPresenceConfidence: 0.4` は標準（0.5）よりも低めの設定です。
  2. 机面スレスレの画角では、机面のテカリや手の濃い影などを「低確信度（0.3〜0.4程度）」で手の一部と誤認しやすくなります。
  3. 本来であれば棄却すべき不安定な推定結果を採用してしまうため、マーカーのプルプルした震えや突発的な座標ズレが発生しやすくなります。

---

### 原因 ⑤：基準手長 `L` の分母計算に「未平滑化の生座標（＋不安定なz座標）」を使用

- **該当コード**: `main.js` 1785〜1797行目
```javascript
// 手首（Landmark 0）と対象指の付け根（baseIdx）間の3D距離 L（手の基準長）
const baseLm = landmarks[fingerConfig.baseIdx];
const wristLm = landmarks[0];
const dx0base = baseLm.x - wristLm.x;
const dy0base = baseLm.y - wristLm.y;
const dz0base = (baseLm.z ?? 0) - (wristLm.z ?? 0);
const L = Math.hypot(dx0base, dy0base, dz0base) || 0.001;

// 平滑化座標に基づく安定した相対変位 ry
const currentRy = (smoothTip.y - smoothBase.y) / (height * L);
```

- **メカニズムと問題点**:
  1. 分子の `smoothTip.y - smoothBase.y` は平滑化されているのに対し、分母の基準長 `L` は生の `landmarks` をそのまま使っています。
  2. さらに、単眼RGBカメラからの深度推定値 `baseLm.z`, `wristLm.z` は平面座標（x, y）に比べて著しくジッター（誤差）が大きい特性があります。
  3. このため、マーカーが飛んでいない場合でも、分母 `L` が1フレームだけ急縮小して `currentRy` がスパイクし、「打鍵していないのに音が鳴る」「判定が暴発する」という間接的なマーカー飛び挙動を招きます。

---

## 3. 具体的な改善策（ロードマップ）

マーカー飛びを根本から防止するための推奨実装案です。

| 改善項目 | 現状の課題 | 改善実装策 |
| :--- | :--- | :--- |
| **1. 全指先ワープ遮断ガード** | 手首（Landmark 0）の移動量しか見ていない | 手首に加えて、各指先端（Landmarks 4, 8, 12, 16, 20）の前フレームからの物理最大移動量（例: 画面幅の8%以上）を超えた場合に、その関節のみ直前座標でクランプまたはホールドする。 |
| **2. 物理移動量リミッター（速度クランプ）** | 1フレームの長距離ワープを高速移動と誤認して追従する | `OneEuroFilter` に入力する前段階で、1フレームあたりの最大変位量（例: 1フレーム最大40px）でクランプし、物理的に不可能な光速ジャンプを遮断する。 |
| **3. ロスト補間時のフィルター継続更新** | ロスト中にフィルターが停止し、復帰初フレームで段差が出る | ロスト中（1〜3フレーム）も直前予測値を用いてフィルターの内部時刻（`tPrev`）と状態を空回し更新し、復帰時の時間差による微分スパイクをゼロにする。 |
| **4. 基準長 `L` の平滑化・2D化** | 分母 `L` に生の3D座標（z誤差大）が混入している | 基準長 `L` の計算にも平滑化済み座標（`smoothedLandmarks`）を使用し、誤差の大きい z 深度を除外して安定した 2D 距離（x, y）基準とする。 |
| **5. 信頼度閾値の適正化** | 0.3〜0.4 でゴースト推定を拾ってしまう | `minTrackingConfidence: 0.5`, `minHandPresenceConfidence: 0.5` に引き上げ、机面の反射や影による誤検出フレームを破棄する。 |

---

## 4. 結論

現状のコードにおけるマーカー飛びは、**「手首しか監視していないジャンプガードの死角」** と **「外れ値に対して平滑化が弱まる 1 Euro Filter の特性」** が主因です。

上記「全指先ワープ遮断ガード」および「物理速度クランプ」を導入することで、演奏中・待機中を問わず、安定した滑らかな骨格描画と確実な打鍵判定が実現できます。
