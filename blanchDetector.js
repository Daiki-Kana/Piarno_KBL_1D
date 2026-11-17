/**
 * BlanchTouch 実験用：指先ROI切り出し ＆ 皮膚白色化（ブランチング）検出モジュール
 *
 * カメラ映像全体ではなく、MediaPipeで検出した対象指の先端（TIP）〜第一関節（DIP/IP）
 * 周辺の小さな矩形領域（ROI）のみをオフスクリーンCanvasへ切り出して解析する。
 * 照明や肌色の個人差を吸収するため、非押下時のベースライン（EMA）からの
 * CIE L*a*b* 相対変化（赤み低下 Δa* と輝度上昇 ΔL*）で押下を判定する。
 */

/**
 * sRGB (0-255) の平均値から CIE L*a*b* (D65) を算出する
 * @param {number} r 0..255
 * @param {number} g 0..255
 * @param {number} b 0..255
 * @returns {{ L: number, a: number, b: number }}
 */
function rgbToLab(r, g, b) {
  // sRGB -> Linear RGB
  let sr = r / 255;
  let sg = g / 255;
  let sb = b / 255;

  sr = sr > 0.04045 ? Math.pow((sr + 0.055) / 1.055, 2.4) : sr / 12.92;
  sg = sg > 0.04045 ? Math.pow((sg + 0.055) / 1.055, 2.4) : sg / 12.92;
  sb = sb > 0.04045 ? Math.pow((sb + 0.055) / 1.055, 2.4) : sb / 12.92;

  // Linear RGB -> XYZ (D65)
  const x = (sr * 0.4124564 + sg * 0.3575761 + sb * 0.1804375) / 0.95047;
  const y = (sr * 0.2126729 + sg * 0.7151522 + sb * 0.0721750) / 1.00000;
  const z = (sr * 0.0193339 + sg * 0.1191920 + sb * 0.9503041) / 1.08883;

  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);

  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz)
  };
}

export class BlanchDetector {
  /**
   * @param {object} [options]
   */
  constructor(options = {}) {
    // チューニング可能パラメータ
    this.roiSize = options.roiSize ?? 24;
    this.onThreshold = options.onThreshold ?? 3.5;
    this.hysteresisRatio = options.hysteresisRatio ?? 0.6;
    this.smoothingAlpha = options.smoothingAlpha ?? 0.35;
    this.baselineAlpha = options.baselineAlpha ?? 0.04;

    // 色変化スコアの重み（赤み低下を主、輝度上昇を補助）
    this.weightA = options.weightA ?? 1.0;
    this.weightL = options.weightL ?? 0.45;

    // 指先〜第一関節距離に対するROI切り出しサイズ倍率とオフセット
    this.roiScaleFactor = options.roiScaleFactor ?? 0.9;
    this.tipOffsetRatio = options.tipOffsetRatio ?? -0.15;

    // 内部オフスクリーンCanvas（ROIサイズ分のみ確保し全画面走査を回避）
    this.roiCanvas = document.createElement("canvas");
    this.roiCanvas.width = this.roiSize;
    this.roiCanvas.height = this.roiSize;
    this.roiCtx = this.roiCanvas.getContext("2d", { willReadFrequently: true });

    // 状態変数
    this.hasBaseline = false;
    this.baseL = 0;
    this.baseA = 0;
    this.smoothScore = 0;
    this.rawScore = 0;
    this.deltaA = 0;
    this.deltaL = 0;
    this.isBlanching = false;
    this.lastRoiRect = null;
  }

  /**
   * 閾値やROIサイズなどの設定を更新
   * @param {object} params
   */
  updateSettings(params = {}) {
    if (typeof params.roiSize === "number" && params.roiSize >= 8 && params.roiSize <= 64) {
      const nextSize = Math.round(params.roiSize);
      if (nextSize !== this.roiSize) {
        this.roiSize = nextSize;
        this.roiCanvas.width = nextSize;
        this.roiCanvas.height = nextSize;
      }
    }
    if (typeof params.onThreshold === "number") {
      this.onThreshold = Math.max(0.1, params.onThreshold);
    }
    if (typeof params.smoothingAlpha === "number") {
      this.smoothingAlpha = Math.min(1.0, Math.max(0.01, params.smoothingAlpha));
    }
    if (typeof params.hysteresisRatio === "number") {
      this.hysteresisRatio = Math.min(0.95, Math.max(0.1, params.hysteresisRatio));
    }
  }

  /**
   * ベースラインおよび平滑化状態をリセット（ターゲット指変更時や手ロスト復帰時）
   */
  resetBaseline() {
    this.hasBaseline = false;
    this.baseL = 0;
    this.baseA = 0;
    this.smoothScore = 0;
    this.rawScore = 0;
    this.deltaA = 0;
    this.deltaL = 0;
    this.isBlanching = false;
  }

  /**
   * 現在のOFF閾値を取得
   * @returns {number}
   */
  getOffThreshold() {
    return this.onThreshold * this.hysteresisRatio;
  }

  /**
   * ビデオフレームから指先ROIを切り出し、ブランチングスコアと判定状態を更新する
   * @param {HTMLVideoElement} video
   * @param {object} tipNorm 先端ランドマーク（正規化座標 0..1）
   * @param {object} dipNorm 第一関節ランドマーク（正規化座標 0..1）
   * @param {boolean} [allowBaselineUpdate=true] 外部ステートがIDLE等のときのみベースライン更新を許可
   * @returns {{
   *   valid: boolean,
   *   isBlanching: boolean,
   *   smoothScore: number,
   *   rawScore: number,
   *   deltaA: number,
   *   deltaL: number,
   *   onThreshold: number,
   *   offThreshold: number
   * }}
   */
  processFrame(video, tipNorm, dipNorm, allowBaselineUpdate = true) {
    const offThreshold = this.getOffThreshold();
    if (!video || !video.videoWidth || !video.videoHeight || !tipNorm || !dipNorm) {
      return {
        valid: false,
        isBlanching: this.isBlanching,
        smoothScore: this.smoothScore,
        rawScore: this.rawScore,
        deltaA: this.deltaA,
        deltaL: this.deltaL,
        onThreshold: this.onThreshold,
        offThreshold
      };
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // 正規化座標をビデオのネイティブピクセル座標へ変換
    const tipX = tipNorm.x * vw;
    const tipY = tipNorm.y * vh;
    const dipX = dipNorm.x * vw;
    const dipY = dipNorm.y * vh;

    // 指先〜第一関節のベクトルと長さからROI中心・サイズを動的に決定
    const dx = tipX - dipX;
    const dy = tipY - dipY;
    const dist = Math.hypot(dx, dy);

    // 切り出し一辺の長さ（ピクセル空間で極端な値にならないようクランプ）
    const boxSize = Math.min(120, Math.max(14, dist * this.roiScaleFactor));

    // 先端からわずかに第一関節側へ寄せた位置を中心にする（背景混入の低減）
    const centerX = tipX + dx * this.tipOffsetRatio;
    const centerY = tipY + dy * this.tipOffsetRatio;

    // ビデオ境界内にクランプした切り出し矩形
    const half = boxSize * 0.5;
    const sx = Math.max(0, Math.min(vw - boxSize, centerX - half));
    const sy = Math.max(0, Math.min(vh - boxSize, centerY - half));
    const sw = Math.min(boxSize, vw - sx);
    const sh = Math.min(boxSize, vh - sy);

    this.lastRoiRect = { sx, sy, sw, sh };

    // 小型オフスクリーンCanvasへROI領域のみを描画・ピクセル取得
    const size = this.roiSize;
    this.roiCtx.drawImage(video, sx, sy, sw, sh, 0, 0, size, size);
    const imgData = this.roiCtx.getImageData(0, 0, size, size).data;

    // 中央円形マスク（四隅の机・鍵盤背景を除外）内の平均RGBを算出
    const cx = (size - 1) * 0.5;
    const cy = (size - 1) * 0.5;
    const radiusSq = (size * 0.42) * (size * 0.42);

    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    let count = 0;

    for (let y = 0; y < size; y++) {
      const py = y - cy;
      const pySq = py * py;
      const rowOffset = y * size;
      for (let x = 0; x < size; x++) {
        const px = x - cx;
        if (px * px + pySq <= radiusSq) {
          const idx = (rowOffset + x) * 4;
          sumR += imgData[idx];
          sumG += imgData[idx + 1];
          sumB += imgData[idx + 2];
          count++;
        }
      }
    }

    if (count === 0) {
      return {
        valid: false,
        isBlanching: this.isBlanching,
        smoothScore: this.smoothScore,
        rawScore: this.rawScore,
        deltaA: this.deltaA,
        deltaL: this.deltaL,
        onThreshold: this.onThreshold,
        offThreshold
      };
    }

    const lab = rgbToLab(sumR / count, sumG / count, sumB / count);

    // 初回フレームでベースラインを即座に初期化
    if (!this.hasBaseline) {
      this.baseL = lab.L;
      this.baseA = lab.a;
      this.hasBaseline = true;
    }

    // 相対変化の算出（赤みの低下 ＋ 輝度の上昇）
    const deltaA = this.baseA - lab.a;
    const deltaL = lab.L - this.baseL;
    const rawScore = this.weightA * Math.max(0, deltaA) + this.weightL * Math.max(0, deltaL);

    // 非押下時かつ白色化が起きていない安定フレームでのみベースラインをゆっくり追従
    if (allowBaselineUpdate && !this.isBlanching && rawScore < this.onThreshold * 0.55) {
      const alpha = this.baselineAlpha;
      this.baseL = alpha * lab.L + (1 - alpha) * this.baseL;
      this.baseA = alpha * lab.a + (1 - alpha) * this.baseA;
    }

    // 時間方向の平滑化（EMA）
    this.smoothScore = this.smoothingAlpha * rawScore + (1 - this.smoothingAlpha) * this.smoothScore;
    this.rawScore = rawScore;
    this.deltaA = deltaA;
    this.deltaL = deltaL;

    // ヒステリシス判定（チャタリング防止）
    if (!this.isBlanching && this.smoothScore >= this.onThreshold) {
      this.isBlanching = true;
    } else if (this.isBlanching && this.smoothScore <= offThreshold) {
      this.isBlanching = false;
    }

    return {
      valid: true,
      isBlanching: this.isBlanching,
      smoothScore: this.smoothScore,
      rawScore: this.rawScore,
      deltaA: this.deltaA,
      deltaL: this.deltaL,
      onThreshold: this.onThreshold,
      offThreshold
    };
  }

  /**
   * デバッグ用プレビューCanvasにROI領域と集計円ガイドを拡大描画する
   * @param {HTMLCanvasElement} previewCanvas
   */
  renderPreview(previewCanvas) {
    if (!previewCanvas) return;
    const ctx = previewCanvas.getContext("2d");
    if (!ctx) return;

    const w = previewCanvas.width;
    const h = previewCanvas.height;
    ctx.clearRect(0, 0, w, h);

    // インカメラ鏡像表示に合わせてプレビューも左右反転して直感性を統一
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.roiCanvas, 0, 0, w, h);
    ctx.restore();

    // 集計領域（中央円形マスク）のガイド表示
    ctx.save();
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.5, w * 0.42, 0, Math.PI * 2);
    ctx.strokeStyle = this.isBlanching ? "rgba(255, 255, 255, 0.95)" : "rgba(255, 255, 255, 0.45)";
    ctx.lineWidth = this.isBlanching ? 2 : 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.restore();
  }
}

