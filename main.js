/**
 * MediaPipe Hand Landmarker 最小検証環境 (インカメラ・鏡像完全一致)
 * 
 * 机面スレスレの水平アングルから手を認識させる前提の最小実装。
 * 遅延検証のため、平滑化フィルター（移動平均や1 Euro Filter等）は一切挟まず、
 * 検出された生（Raw）のランドマーク座標をダイレクトにCanvasへ描画します。
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// DOM要素
const video = document.getElementById("webcam");
const canvas = document.getElementById("output-canvas");
const canvasCtx = canvas.getContext("2d");
const statusText = document.getElementById("status-text");
const fpsCounter = document.getElementById("fps-counter");
const inferenceTime = document.getElementById("inference-time");
const handsCount = document.getElementById("hands-count");
const camInfo = document.getElementById("cam-info");
const errorBox = document.getElementById("error-box");
const tipXVal = document.getElementById("tip-x-val");
const tipYVal = document.getElementById("tip-y-val");
const relYVal = document.getElementById("rel-y-val");
const thVal = document.getElementById("th-val");
const stateVal = document.getElementById("state-val");
const tapCountVal = document.getElementById("tap-count-val");
const csvStatus = document.getElementById("csv-status");
const csvFileInput = document.getElementById("csv-file-input");
const testSoundBtn = document.getElementById("test-sound-btn");
const targetFingerVal = document.getElementById("target-finger-val");
const targetTapProgress = document.getElementById("target-tap-progress");

// テスト用CSVデータセット群（ファイル別）
export let testDataDatasets = [];
// 全ファイルのフレームを結合したフラット配列
export let testDataFrames = [];

// Web Audio API サウンドエンジン
let audioCtx = null;
let tapCount = 0;

// 学習済みCSVから自動導出される閾値モデル（デフォルト値付き）
let hitRyThreshold = 0.70;
let liftRyThreshold = 0.45;
let trainedHitSamples = 0;

// 打鍵ステートマシン管理（IDLE: 待機, TOUCHED: 机面接触中・リフト待ち）
let tapState = "IDLE";

/**
 * AudioContextの初期化・再開（ブラウザのAutoplay Policy対応）
 */
function ensureAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().then(() => {
      console.log("[AUDIO] AudioContext resumed successfully (running)");
    }).catch((err) => {
      console.warn("[AUDIO] AudioContext resume failed:", err);
    });
  }
  return audioCtx;
}

/**
 * ゼロ遅延の打鍵音生成（指数減衰ピアノ風トーン）
 * @param {number} freq 周波数 (デフォルト: 523.25Hz = C5)
 */
export function playTapSound(freq = 523.25) {
  const ctx = ensureAudioContext();
  if (!ctx) return;

  if (ctx.state === "suspended") {
    ctx.resume();
  }

  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  // トライアングル波（ピアノに似た豊かな倍音成分）
  osc.type = "triangle";
  osc.frequency.setValueAtTime(freq, now);

  // 明瞭なエンベロープ（3msで立ち上がり、約0.18秒で自然に減衰）
  gain.gain.setValueAtTime(0.001, now);
  gain.gain.linearRampToValueAtTime(0.50, now + 0.003);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(now + 0.19);

  // カウント更新
  tapCount++;
  if (tapCountVal) {
    tapCountVal.textContent = `${tapCount}`;
  }
}

// 状態管理
let handLandmarker = null;
let currentStream = null;
let isPredicting = false;
let lastVideoTime = -1;
let lastFpsUpdateTime = performance.now();
let frameCount = 0;

// 人差し指先端（TIP）の最新生ピクセル座標
export const currentTip = {
  x: 0,
  y: 0
};

/**
 * 1 Euro Filter（ワンユーロフィルター）
 * 低速時はジッター（微小な震え）を完全に平滑化し、高速打鍵時は遅延ゼロで追従する標準適応フィルター
 */
class OneEuroFilter {
  constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, timestamp = performance.now()) {
    if (this.xPrev === null || this.tPrev === null) {
      this.xPrev = x;
      this.dxPrev = 0;
      this.tPrev = timestamp;
      return x;
    }

    const dt = Math.max((timestamp - this.tPrev) / 1000, 0.001);
    this.tPrev = timestamp;

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
    this.xPrev = xHat;

    return xHat;
  }
}

/**
 * 2D関節座標用 1 Euro Filter
 */
class PointFilter {
  constructor(minCutoff = 1.2, beta = 0.008) {
    this.xf = new OneEuroFilter(minCutoff, beta);
    this.yf = new OneEuroFilter(minCutoff, beta);
  }

  reset() {
    this.xf.reset();
    this.yf.reset();
  }

  filter(x, y, timestamp) {
    return {
      x: this.xf.filter(x, timestamp),
      y: this.yf.filter(y, timestamp)
    };
  }
}

// 対象指の定義（親指〜小指までの全指）
export const FINGER_CONFIGS = {
  THUMB: {
    key: "THUMB",
    label: "親指 (THUMB)",
    freq: 523.25, // C5
    baseIdx: 2,   // MCP (付け根)
    p1Idx: 1,     // CMC
    p2Idx: 2,     // MCP
    p3Idx: 3,     // IP
    tipIdx: 4,    // TIP
    indices: [1, 2, 3, 4]
  },
  INDEX: {
    key: "INDEX",
    label: "人差し指 (INDEX)",
    freq: 587.33, // D5
    baseIdx: 5,   // MCP
    p1Idx: 5,     // MCP
    p2Idx: 6,     // PIP
    p3Idx: 7,     // DIP
    tipIdx: 8,    // TIP
    indices: [5, 6, 7, 8]
  },
  MIDDLE: {
    key: "MIDDLE",
    label: "中指 (MIDDLE)",
    freq: 659.25, // E5
    baseIdx: 9,   // MCP
    p1Idx: 9,     // MCP
    p2Idx: 10,    // PIP
    p3Idx: 11,    // DIP
    tipIdx: 12,   // TIP
    indices: [9, 10, 11, 12]
  },
  RING: {
    key: "RING",
    label: "薬指 (RING)",
    freq: 783.99, // G5
    baseIdx: 13,  // MCP
    p1Idx: 13,    // MCP
    p2Idx: 14,    // PIP
    p3Idx: 15,    // DIP
    tipIdx: 16,   // TIP
    indices: [13, 14, 15, 16]
  },
  PINKY: {
    key: "PINKY",
    label: "小指 (PINKY)",
    freq: 1046.50, // C6
    baseIdx: 17,  // MCP
    p1Idx: 17,    // MCP
    p2Idx: 18,    // PIP
    p3Idx: 19,    // DIP
    tipIdx: 20,   // TIP
    indices: [17, 18, 19, 20]
  }
};

// 親指から順番に2回ずつ自動選択されるシーケンス定義（親指→人差し指→中指→薬指→小指）
export const FINGER_SEQUENCE = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"];
export const TAPS_PER_FINGER = 2;

export let currentSeqIndex = 0;
export let currentFingerTaps = 0;
export let currentFingerKey = FINGER_SEQUENCE[0]; // 初期: "THUMB"

// 選択中の指4点専用の適応フィルター
const targetPointFilters = {
  p1: new PointFilter(1.5, 0.01),
  p2: new PointFilter(1.5, 0.01),
  p3: new PointFilter(1.5, 0.01),
  tip: new PointFilter(1.5, 0.01)
};

// 指ごとの学習閾値モデル
export const fingerThresholdModels = {
  THUMB: { hitRy: 0.80, liftRy: 0.55, samples: 0 },
  MIDDLE: { hitRy: 0.80, liftRy: 0.55, samples: 0 },
  RING: { hitRy: 0.80, liftRy: 0.55, samples: 0 },
  PINKY: { hitRy: 0.80, liftRy: 0.55, samples: 0 },
  INDEX: { hitRy: 0.80, liftRy: 0.55, samples: 0 }
};

// 指ごとのデータセット格納用
export const fingerDatasets = {
  THUMB: [],
  MIDDLE: [],
  RING: [],
  PINKY: [],
  INDEX: []
};

/**
 * ターゲット指の変更と個別閾値の適用
 * @param {string} fingerKey
 */
export function setTargetFinger(fingerKey) {
  if (!FINGER_CONFIGS[fingerKey]) return;
  currentFingerKey = fingerKey;
  targetPointFilters.p1.reset();
  targetPointFilters.p2.reset();
  targetPointFilters.p3.reset();
  targetPointFilters.tip.reset();

  // 指別学習モデルの閾値を適用
  applyFingerThreshold(fingerKey);

  // HUD表示更新
  if (targetFingerVal) {
    targetFingerVal.textContent = fingerKey;
    targetFingerVal.classList.add("spike-highlight");
    setTimeout(() => {
      if (targetFingerVal) targetFingerVal.classList.remove("spike-highlight");
    }, 200);
  }
  updateTapProgressHud();

  console.log(`[FINGER SEQ] ターゲット指を変更: ${FINGER_CONFIGS[fingerKey].label} (TH: ${hitRyThreshold.toFixed(2)})`);
}

/**
 * 指別打鍵進捗HUDの更新
 */
export function updateTapProgressHud() {
  if (targetTapProgress) {
    targetTapProgress.textContent = `${currentFingerTaps} / ${TAPS_PER_FINGER}`;
  }
}

/**
 * 指定指の学習済み閾値をアクティブ閾値にセット
 * @param {string} fingerKey
 */
export function applyFingerThreshold(fingerKey) {
  const model = fingerThresholdModels[fingerKey];
  if (model) {
    hitRyThreshold = model.hitRy;
    liftRyThreshold = model.liftRy;
    trainedHitSamples = model.samples;
  }
  if (thVal) {
    thVal.textContent = hitRyThreshold.toFixed(2);
  }
}

// 手の骨格コネクション定義（全21ランドマーク間の接続）
const HAND_CONNECTIONS = [
  // 手のひら
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  // 親指
  [1, 2], [2, 3], [3, 4],
  // 人差し指（背景薄表示）
  [5, 6], [6, 7], [7, 8],
  // 中指
  [9, 10], [10, 11], [11, 12],
  // 薬指
  [13, 14], [14, 15], [15, 16],
  // 小指
  [17, 18], [18, 19], [19, 20]
];

/**
 * CSVステータス表示の更新
 * @param {string} msg
 */
function updateCsvStatus(msg) {
  if (csvStatus) {
    csvStatus.textContent = msg;
  }
}

/**
 * CSVテキストをパースして相対変位ryおよび打鍵フラグを抽出
 * （timestamp,rx,ry,rz,vx,vy,vz,label 形式、および frame,timestamp,mcp_x... 形式の両方に対応）
 * @param {string} text CSVテキスト
 * @returns {Array<{ timestamp: number, ry: number, isHit: boolean }>}
 */
/**
 * CSVテキストをパースして相対変位ryおよび打鍵フラグを抽出
 * （打鍵ラベルが未付与のtapファイルの場合、変位ピークを自動検出して打鍵点として認識）
 * @param {string} text CSVテキスト
 * @param {string} filename ファイル名
 * @returns {Array<{ timestamp: number, ry: number, isHit: boolean }>}
 */
export function parseCsv(text, filename = "") {
  if (!text || typeof text !== "string") return [];
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const headerLine = lines[0].toLowerCase();
  const headers = headerLine.split(",").map((h) => h.trim());

  const ryIdx = headers.indexOf("ry");
  const labelIdx = headers.indexOf("label") !== -1 ? headers.indexOf("label") : headers.indexOf("is_hit");
  const timeIdx = headers.indexOf("timestamp") !== -1 ? headers.indexOf("timestamp") : (headers.indexOf("timestamp_ms") !== -1 ? headers.indexOf("timestamp_ms") : headers.indexOf("time_sec"));

  const mcpYIdx = headers.indexOf("index_mcp_y") !== -1 ? headers.indexOf("index_mcp_y") : headers.indexOf("mcp_y");
  const tipYIdx = headers.indexOf("index_tip_y") !== -1 ? headers.indexOf("index_tip_y") : headers.indexOf("tip_y");

  const results = [];
  let manualHitCount = 0;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(",").map((c) => c.trim());

    let ry = null;
    let isHit = false;
    let timestamp = 0;

    if (timeIdx !== -1 && cols[timeIdx]) {
      timestamp = parseFloat(cols[timeIdx]) || 0;
    }

    if (labelIdx !== -1 && cols[labelIdx]) {
      isHit = parseInt(cols[labelIdx], 10) === 1;
      if (isHit) manualHitCount++;
    }

    if (ryIdx !== -1 && cols[ryIdx] !== undefined && cols[ryIdx] !== "") {
      ry = parseFloat(cols[ryIdx]);
    } else if (mcpYIdx !== -1 && tipYIdx !== -1 && cols[mcpYIdx] && cols[tipYIdx]) {
      const my = parseFloat(cols[mcpYIdx]);
      const ty = parseFloat(cols[tipYIdx]);
      if (!isNaN(my) && !isNaN(ty)) {
        ry = ty - my;
      }
    }

    if (ry !== null && !isNaN(ry)) {
      results.push({
        timestamp,
        ry,
        isHit
      });
    }
  }

  // もしファイル名に "tap" が含まれており、手動打鍵マークが0件の場合は自動ピーク検出を実行
  if (filename.toLowerCase().includes("tap") && manualHitCount === 0 && results.length > 10) {
    autoDetectTapPeaks(results);
  }

  return results;
}

/**
 * 打鍵データ波形から机面接触ピーク（極大点）を自動検出して isHit を付与
 * @param {Array<{ timestamp: number, ry: number, isHit: boolean }>} frames
 */
function autoDetectTapPeaks(frames) {
  const rys = frames.map((f) => f.ry);
  const len = rys.length;
  if (len < 5) return;

  // 全体の平均値と中央値を算出
  const sortedRys = [...rys].sort((a, b) => a - b);
  const medianRy = sortedRys[Math.floor(len * 0.5)];
  const p75Ry = sortedRys[Math.floor(len * 0.75)];

  let lastHitIdx = -100;
  let autoCount = 0;

  for (let i = 3; i < len - 3; i++) {
    const cur = rys[i];
    // 局所極大（前後より高く、かつ75%タイル以上の山）
    if (
      cur > rys[i - 1] &&
      cur >= rys[i + 1] &&
      cur > rys[i - 2] &&
      cur >= rys[i + 2] &&
      cur > p75Ry &&
      (cur - medianRy) > 0.08 &&
      (i - lastHitIdx) > 10 // クールダウン（最低10フレーム離れている）
    ) {
      frames[i].isHit = true;
      lastHitIdx = i;
      autoCount++;
    }
  }

  console.log(`[CSV PEAK] 打鍵ピーク自動検出: ${autoCount} 箇所の打鍵点を抽出 (Median=${medianRy.toFixed(2)}, P75=${p75Ry.toFixed(2)})`);
}

/**
 * 読み込んだ学習済みCSVデータから各指の打鍵判定閾値を自動導出
 */
function trainModelFromCsv() {
  const fingerKeys = ["THUMB", "MIDDLE", "RING", "PINKY", "INDEX"];
  for (const fKey of fingerKeys) {
    trainModelForFinger(fKey);
  }
  // 現在選択されている指の閾値を反映
  applyFingerThreshold(currentFingerKey);
}

/**
 * 特定指のデータセットから打鍵閾値を導出（負値変位・各指の可動域に完全対応）
 * @param {string} fingerKey
 */
function trainModelForFinger(fingerKey) {
  const datasets = fingerDatasets[fingerKey] || [];
  if (datasets.length === 0) return;

  const frames = datasets.flatMap((d) => d.frames);
  if (frames.length === 0) return;

  // 1. 打鍵データ（isHit = true）の抽出
  const hitRys = frames
    .filter((f) => f.isHit && f.ry !== null && !isNaN(f.ry))
    .map((f) => f.ry)
    .sort((a, b) => a - b);

  // 2. 空中動作データ（純粋な空中浮遊データ neg_air のみを使用）
  // ※ neg_idle (静止待機) や neg_slide (スライド) は机面接触を含むため空中基準から除外
  const airDatasets = datasets.filter((d) => {
    const fn = d.filename.toLowerCase();
    return fn.includes("neg_air");
  });

  let maxAirRy = -Infinity;
  if (airDatasets.length > 0) {
    const airFrames = airDatasets.flatMap((d) => d.frames);
    const validAirRys = airFrames
      .filter((f) => f.ry !== null && !isNaN(f.ry))
      .map((f) => f.ry)
      .sort((a, b) => a - b);
    if (validAirRys.length > 0) {
      // 90パーセンタイルを空中最大変位として採用（異常外れ値をカット）
      maxAirRy = validAirRys[Math.floor(validAirRys.length * 0.90)];
    }
  }

  // 空中データが無い場合は打鍵以外のデータの下位70%を参照
  if (maxAirRy === -Infinity) {
    const negRys = frames
      .filter((f) => !f.isHit && f.ry !== null && !isNaN(f.ry))
      .map((f) => f.ry)
      .sort((a, b) => a - b);
    maxAirRy = negRys.length > 0 ? negRys[Math.floor(negRys.length * 0.70)] : 0.0;
  }

  const samples = hitRys.length;
  let hitTh = 0.0;
  let liftTh = 0.0;

  if (samples > 0) {
    // 打鍵サンプルの中央値・第25%パーセンタイル値
    const medianHitRy = hitRys[Math.floor(samples * 0.50)];
    const p25HitRy = hitRys[Math.floor(samples * 0.25)];

    if (medianHitRy > maxAirRy) {
      // 空中最大と打鍵中央値の中間点に打鍵閾値を設定
      // 薬指（RING）は可動域が狭いため打鍵マージンをやや緩和（0.38）して叩きやすくする
      const hitRatio = fingerKey === "RING" ? 0.38 : 0.45;
      hitTh = maxAirRy + (medianHitRy - maxAirRy) * hitRatio;

      // リフト閾値：全指で打鍵位置からわずかに指を浮かせるだけで素早くIDLE復帰できるよう、
      // ヒステリシス幅を極小（0.03）に設定
      const hysteresis = 0.03;
      liftTh = hitTh - hysteresis;
    } else {
      // 外れ値等で打鍵中央値が空中を下回る場合の適応フォールバック
      hitTh = p25HitRy;
      liftTh = hitTh - 0.03;
    }
  } else {
    // 打鍵サンプルが0件の場合の安全マージン
    hitTh = maxAirRy + 0.18;
    liftTh = hitTh - 0.03;
  }

  fingerThresholdModels[fingerKey] = {
    hitRy: parseFloat(hitTh.toFixed(2)),
    liftRy: parseFloat(liftTh.toFixed(2)),
    samples
  };

  console.log(
    `[MODEL: ${fingerKey}] 学習完了: 打鍵=${samples}件 (空中Max=${maxAirRy.toFixed(3)}) -> 打鍵TH=${hitTh.toFixed(2)}, リフトTH=${liftTh.toFixed(2)}`
  );
}


/**
 * 複数CSVテキストを一括パースしてデータセット配列に格納
 * @param {Array<{ name: string, content: string }>} files
 */
function ingestCsvFiles(files) {
  for (const file of files) {
    const parsed = parseCsv(file.content, file.name);
    if (parsed.length > 0) {
      const existIdx = testDataDatasets.findIndex((d) => d.filename === file.name);
      const datasetEntry = {
        filename: file.name,
        frames: parsed,
        hitCount: parsed.filter((f) => f.isHit).length
      };

      if (existIdx !== -1) {
        testDataDatasets[existIdx] = datasetEntry;
      } else {
        testDataDatasets.push(datasetEntry);
      }
    }
  }

  // 全フレームのフラット配列を生成
  testDataFrames = testDataDatasets.flatMap((d) => d.frames);

  if (testDataDatasets.length > 0) {
    const totalHits = testDataFrames.filter((f) => f.isHit).length;
    trainModelFromCsv();
    updateCsvStatus(`LOADED (${testDataDatasets.length} files, ${totalHits} hits)`);
    console.log(
      `[CSV] 全読込成功: 計 ${testDataDatasets.length} ファイル, ${testDataFrames.length} フレーム (打鍵マーク: ${totalHits})`
    );
  } else {
    updateCsvStatus("NOT LOADED");
    if (thVal) thVal.textContent = hitRyThreshold.toFixed(2);
  }
}

/**
 * ファイルパスまたはファイル名から指キー（THUMB, MIDDLE, RING, PINKY, INDEX）を判定
 * @param {string} path
 * @returns {string}
 */
export function detectFingerKey(path) {
  const p = path.toLowerCase();
  if (p.includes("thumb")) return "THUMB";
  if (p.includes("middle")) return "MIDDLE";
  if (p.includes("ring")) return "RING";
  if (p.includes("pinky")) return "PINKY";
  if (p.includes("index")) return "INDEX";
  return "THUMB";
}

/**
 * public/dataset/ 配下の各指CSVデータセットを自動非同期フェッチして読み込む
 */
export async function loadTestDataCsv() {
  const loadedFiles = [];

  // 初期化
  Object.keys(fingerDatasets).forEach((k) => (fingerDatasets[k] = []));

  try {
    const manifestRes = await fetch("./dataset/manifest.json");
    if (manifestRes.ok) {
      const manifestData = await manifestRes.json();
      let fileList = [];
      if (Array.isArray(manifestData)) {
        fileList = manifestData;
      } else if (manifestData && typeof manifestData === "object") {
        if (manifestData.fingers) {
          fileList = Object.values(manifestData.fingers).flat();
        } else if (Array.isArray(manifestData.files)) {
          fileList = manifestData.files;
        }
      }

      for (const fname of fileList) {
        if (!fname) continue;
        try {
          const csvRes = await fetch(`./dataset/${fname}`);
          if (csvRes.ok) {
            const text = await csvRes.text();
            const displayName = fname.includes("/") ? fname.split("/").pop() : fname;
            const fingerKey = detectFingerKey(fname);
            const parsed = parseCsv(text, displayName);
            if (parsed.length > 0) {
              const entry = {
                filename: displayName,
                path: fname,
                fingerKey,
                frames: parsed,
                hitCount: parsed.filter((f) => f.isHit).length
              };
              fingerDatasets[fingerKey].push(entry);
              loadedFiles.push({ name: displayName, content: text });
            }
          }
        } catch (err) {
          console.warn(`[CSV] ${fname} の取得スキップ:`, err);
        }
      }
    }
  } catch (err) {
    console.warn("[CSV] manifest.json の読み込みをスキップ:", err);
  }

  if (loadedFiles.length > 0) {
    ingestCsvFiles(loadedFiles);
    trainModelFromCsv();
  } else {
    updateCsvStatus("NO CSV DATA");
    applyFingerThreshold(currentFingerKey);
  }
}


/**
 * 初期化処理
 */
async function init() {
  resetDebugMetrics();
  updateStatus("モデル読込中...");

  // テスト用CSVの非同期読み込み（未配置でもブロックせず実行）
  loadTestDataCsv();

  try {
    // 1. MediaPipe Hand Landmarker の読み込み
    await initHandLandmarker();

    // 2. インカメラ（フロントカメラ）の自動起動
    updateStatus("インカメラ起動中...");
    await startFrontCamera();
  } catch (error) {
    console.error("初期化エラー:", error);
    showError(error.message || "初期化に失敗しました");
    updateStatus("エラー停止");
  }
}

/**
 * MediaPipe Hand Landmarker の初期化
 */
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const modelPaths = [
    "./models/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  ];

  let loaded = false;
  let lastErr = null;

  for (const modelPath of modelPaths) {
    try {
      // 60fps安定化のため検出対象手を1つに限定して推論時間を最小化
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      loaded = true;
      break;
    } catch (gpuErr) {
      console.warn("GPU失敗、CPUへフォールバック:", gpuErr);
      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: modelPath,
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        loaded = true;
        break;
      } catch (err) {
        lastErr = err;
      }
    }
  }

  if (!loaded) {
    throw new Error(`モデルロード失敗: ${lastErr?.message || "不明なエラー"}`);
  }
}

/**
 * getUserMedia 互換処理
 * @param {MediaStreamConstraints} constraints
 */
function getCompatibleUserMedia(constraints) {
  if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  if (!window.isSecureContext) {
    throw new Error(
      "非セキュア環境 (HTTP) のためカメラがブロックされています。HTTPS (https://...) でアクセスしてください。"
    );
  }

  const legacyGetUserMedia =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

  if (legacyGetUserMedia) {
    return new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, constraints, resolve, reject);
    });
  }

  throw new Error("このブラウザはカメラAPIに対応していません。");
}

/**
 * インカメラ（フロントカメラ）の自動起動
 */
async function startFrontCamera() {
  let stream = null;

  // 16:9の自然な画角比率を維持しつつ、60fps出力（720p / 540p / 360p）を優先探索
  const tryConstraints = [
    // 1. 720p (1280x720, 16:9) 60fps
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 60, min: 45 }
      },
      audio: false
    },
    // 2. 540p (960x540, 16:9) 60fps
    {
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 60, min: 45 }
      },
      audio: false
    },
    // 3. 360p (640x360, 16:9) 60fps
    {
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 360 },
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 60, min: 45 }
      },
      audio: false
    },
    // 4. 720p (1280x720, 16:9) 任意fps
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 60 }
      },
      audio: false
    },
    // 5. 16:9 比率優先（解像度任意）
    {
      video: {
        facingMode: "user",
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 60 }
      },
      audio: false
    },
    {
      video: true,
      audio: false
    }
  ];

  for (const constraints of tryConstraints) {
    try {
      stream = await getCompatibleUserMedia(constraints);
      if (stream) break;
    } catch (e) {
      console.warn("制約でのカメラ起動失敗、次を試行:", e);
    }
  }

  if (!stream) {
    throw new Error("インカメラの取得に失敗しました。カメラのアクセス許可を確認してください。");
  }

  currentStream = stream;
  video.srcObject = currentStream;

  // ハードウェアがサポートする最大フレームレートを強制適用
  const track = stream.getVideoTracks()[0];
  if (track) {
    if (typeof track.getCapabilities === "function") {
      const caps = track.getCapabilities();
      console.log("[CAM CAPABILITIES]", caps);
      const targetFps = (caps.frameRate && caps.frameRate.max) ? Math.min(60, caps.frameRate.max) : 60;
      try {
        await track.applyConstraints({
          frameRate: { ideal: targetFps }
        });
      } catch (err) {
        console.warn("FPS制約適用スキップ:", err);
      }
    }

    // 実効カメラ設定（解像度・fps）をHUDに反映
    if (typeof track.getSettings === "function") {
      const settings = track.getSettings();
      const fpsLabel = settings.frameRate ? `${Math.round(settings.frameRate)}fps` : "60fps";
      const resLabel = settings.height ? `${settings.height}p` : "-";
      camInfo.textContent = `FRONT (${resLabel} ${fpsLabel})`.trim();
    }
  }

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });

  // Canvas内部解像度をビデオ解像度に完全一致させる
  updateCanvasResolution();

  // 推論ループ開始
  isPredicting = true;
  schedulePredictLoop();

  updateStatus("トラッキング中");
}

/**
 * Canvas内部解像度をビデオのネイティブピクセル数に同期
 */
function updateCanvasResolution() {
  if (video.videoWidth && video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

/**
 * 毎フレームの推論処理（単一フレーム）
 */
function processVideoFrame() {
  if (!isPredicting) return;

  const startTime = performance.now();

  // VIDEOモードでの同期推論
  const results = handLandmarker.detectForVideo(video, startTime);

  const calcDuration = performance.now() - startTime;
  inferenceTime.textContent = `${calcDuration.toFixed(1)} ms`;

  // 生ランドマーク座標の直接描画（平滑化フィルターなし）
  drawRawHandLandmarks(results);

  // FPS更新
  updateFps();
}

/**
 * requestVideoFrameCallback または requestAnimationFrame による推論ループ
 * （カメラの物理フレーム到着に直接同期して60fpsを安定維持）
 */
function schedulePredictLoop() {
  if (!isPredicting) return;

  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(() => {
      processVideoFrame();
      schedulePredictLoop();
    });
  } else {
    requestAnimationFrame(() => {
      if (video.currentTime !== lastVideoTime && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        lastVideoTime = video.currentTime;
        processVideoFrame();
      }
      schedulePredictLoop();
    });
  }
}

/**
 * 人差し指フォーカス描画、打鍵・リフト用特徴量の算出
 * 平滑化フィルターや座標固定クリップは挟まず、Raw座標にダイレクト追従します
 * @param {object} results
 */
function drawRawHandLandmarks(results) {
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  const fingerConfig = FINGER_CONFIGS[currentFingerKey] || FINGER_CONFIGS.THUMB;
  const targetIndices = fingerConfig.indices;

  if (!results || !results.landmarks || results.landmarks.length === 0) {
    handsCount.textContent = "0";
    if (tapState !== "IDLE") {
      tapState = "IDLE";
      updateStateHud("IDLE", false);
    }
    // 手ロスト時にフィルターをリセット（再検出時の座標ジャンプ防止）
    targetPointFilters.p1.reset();
    targetPointFilters.p2.reset();
    targetPointFilters.p3.reset();
    targetPointFilters.tip.reset();
    resetDebugMetrics();
    return;
  }

  handsCount.textContent = `${results.landmarks.length}`;

  const width = canvas.width;
  const height = canvas.height;

  // メインの手（先頭の手）
  const landmarks = results.landmarks[0];

  // 1. 対象指以外の骨格（人差し指を含むすべて）を薄いグレースケールで描画
  const otherConnections = HAND_CONNECTIONS.filter(
    ([s, e]) => !(targetIndices.includes(s) && targetIndices.includes(e))
  );

  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  canvasCtx.lineWidth = 1;
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  otherConnections.forEach(([startIdx, endIdx]) => {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];

    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * width, p1.y * height);
    canvasCtx.lineTo(p2.x * width, p2.y * height);
    canvasCtx.stroke();
  });

  // 対象指以外の関節点（人差し指を含む・極小薄グレー）
  landmarks.forEach((p, idx) => {
    if (targetIndices.includes(idx)) return;
    canvasCtx.beginPath();
    canvasCtx.arc(p.x * width, p.y * height, 2, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "rgba(255, 255, 255, 0.15)";
    canvasCtx.fill();
  });

  // 2手目以降が存在する場合も薄いグレースケールで描画
  if (results.landmarks.length > 1) {
    for (let i = 1; i < results.landmarks.length; i++) {
      const extra = results.landmarks[i];
      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const p1 = extra[startIdx];
        const p2 = extra[endIdx];
        canvasCtx.beginPath();
        canvasCtx.moveTo(p1.x * width, p1.y * height);
        canvasCtx.lineTo(p2.x * width, p2.y * height);
        canvasCtx.stroke();
      });
    }
  }

  // 2. 選択中指ランドマークの生ピクセル座標
  const rawP1 = { x: landmarks[fingerConfig.p1Idx].x * width, y: landmarks[fingerConfig.p1Idx].y * height };
  const rawP2 = { x: landmarks[fingerConfig.p2Idx].x * width, y: landmarks[fingerConfig.p2Idx].y * height };
  const rawP3 = { x: landmarks[fingerConfig.p3Idx].x * width, y: landmarks[fingerConfig.p3Idx].y * height };
  const rawTip = { x: landmarks[fingerConfig.tipIdx].x * width, y: landmarks[fingerConfig.tipIdx].y * height };

  // 1 Euro Filterによる適応平滑化（静止時はジッター完全除去、打鍵時は遅延ゼロ追従）
  const now = performance.now();
  const smoothP1 = targetPointFilters.p1.filter(rawP1.x, rawP1.y, now);
  const smoothP2 = targetPointFilters.p2.filter(rawP2.x, rawP2.y, now);
  const smoothP3 = targetPointFilters.p3.filter(rawP3.x, rawP3.y, now);
  const smoothTip = targetPointFilters.tip.filter(rawTip.x, rawTip.y, now);

  // 最新平滑化座標の保持
  currentTip.x = smoothTip.x;
  currentTip.y = smoothTip.y;

  // 手首（Landmark 0: Wrist）と対象指の付け根（baseIdx）間の3D距離 L（手の基準長）
  const baseLm = landmarks[fingerConfig.baseIdx];
  const dx0base = baseLm.x - landmarks[0].x;
  const dy0base = baseLm.y - landmarks[0].y;
  const dz0base = (baseLm.z ?? 0) - (landmarks[0].z ?? 0);
  const L = Math.hypot(dx0base, dy0base, dz0base) || 0.001;

  // 対象指の基準点ピクセル座標（smoothP2 または smoothP1）
  const smoothBase = (fingerConfig.baseIdx === fingerConfig.p1Idx) ? smoothP1 : smoothP2;

  // 平滑化座標に基づく安定した相対変位 ry
  const currentRy = (smoothTip.y - smoothBase.y) / (height * L);
  const rawRy = (landmarks[fingerConfig.tipIdx].y - baseLm.y) / L;

  // 3. 学習済みCSVモデルによるリアルタイム打鍵認識（空中誤検知を遮断）
  if (tapState === "IDLE") {
    // 平滑化変位が学習された打鍵閾値以上になったら打鍵判定
    if (currentRy >= hitRyThreshold) {
      tapState = "TOUCHED";
      playTapSound(fingerConfig.freq || 523.25); // 指ごとの音階で発音
      updateStateHud("TOUCHED", true);

      // 親指から順に2回ずつ自動カウント
      currentFingerTaps++;
      console.log(`[REALTIME TAP] #${tapCount} [${currentFingerKey}] (${currentFingerTaps}/${TAPS_PER_FINGER}) ry=${currentRy.toFixed(3)} >= TH:${hitRyThreshold.toFixed(2)}`);

      if (currentFingerTaps >= TAPS_PER_FINGER) {
        // 2回打鍵完了！次の指へ自動遷移
        currentFingerTaps = 0;
        currentSeqIndex = (currentSeqIndex + 1) % FINGER_SEQUENCE.length;
        const nextFinger = FINGER_SEQUENCE[currentSeqIndex];
        setTargetFinger(nextFinger);
      } else {
        updateTapProgressHud();
      }
    }
  } else if (tapState === "TOUCHED") {
    // 指のリフト復帰（閾値を下回ったら待機状態へ）
    if (currentRy <= liftRyThreshold) {
      tapState = "IDLE";
      updateStateHud("IDLE", false);
    }
  }

  // 4. デバッグHUDのリアルタイム表示更新（平滑化座標と相対変位）
  updateDebugMetrics(smoothTip.x, smoothTip.y, currentRy);

  // 5. 対象指の骨格描画（控えめで繊細な平滑化ライン）
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  canvasCtx.lineWidth = 1.5;
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  canvasCtx.beginPath();
  canvasCtx.moveTo(smoothP1.x, smoothP1.y);
  canvasCtx.lineTo(smoothP2.x, smoothP2.y);
  canvasCtx.lineTo(smoothP3.x, smoothP3.y);
  canvasCtx.lineTo(smoothTip.x, smoothTip.y);
  canvasCtx.stroke();

  // 対象指関節点（P1, P2, P3）の描画（控えめな小丸）
  [smoothP1, smoothP2, smoothP3].forEach((pt) => {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fill();
    canvasCtx.strokeStyle = "rgba(0, 0, 0, 0.6)";
    canvasCtx.lineWidth = 0.8;
    canvasCtx.stroke();
  });

  // 6. 対象指先端（TIP）の控えめなミニマルマーカー描画
  drawTipTargetMark(smoothTip.x, smoothTip.y);
}

/**
 * 対象指先端（TIP）のミニマルターゲットマーク描画（洗練された極細リング＋中心ドット）
 * @param {number} x
 * @param {number} y
 */
function drawTipTargetMark(x, y) {
  // 外側の繊細なサークル（半径5px、極細線）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  canvasCtx.lineWidth = 1.2;
  canvasCtx.stroke();

  // 中心ミニマムドット（半径2px）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 2, 0, 2 * Math.PI);
  canvasCtx.fillStyle = "#ffffff";
  canvasCtx.fill();
}

/**
 * 人差し指TIPの生座標（X, Y）および相対変位のHUD更新表示（白黒ミニマル）
 * @param {number} x
 * @param {number} y
 * @param {number} ry
 */
function updateDebugMetrics(x, y, ry) {
  tipXVal.textContent = `${x.toFixed(1)}px`;
  tipYVal.textContent = `${y.toFixed(1)}px`;
  if (relYVal) {
    relYVal.textContent = ry !== undefined && ry !== null ? (ry >= 0 ? `+${ry.toFixed(2)}` : ry.toFixed(2)) : "-";
  }
}

/**
 * 手ロスト時のHUDメトリクスリセット
 */
function resetDebugMetrics() {
  tipXVal.textContent = "-";
  tipYVal.textContent = "-";
  if (relYVal) {
    relYVal.textContent = "-";
  }
}

/**
 * 打鍵ステート（IDLE / TOUCHED）のHUD更新と打鍵時ハイライト
 * @param {string} state
 * @param {boolean} isHitFlash
 */
function updateStateHud(state, isHitFlash) {
  if (!stateVal) return;
  stateVal.textContent = state;
  if (isHitFlash) {
    stateVal.classList.add("spike-highlight");
    setTimeout(() => {
      if (stateVal && tapState === "TOUCHED") {
        stateVal.classList.remove("spike-highlight");
      }
    }, 120);
  } else {
    stateVal.classList.remove("spike-highlight");
  }
}

/**
 * FPSカウンターの計算
 */
function updateFps() {
  frameCount++;
  const now = performance.now();
  const elapsed = now - lastFpsUpdateTime;

  if (elapsed >= 500) {
    const fps = ((frameCount * 1000) / elapsed).toFixed(1);
    fpsCounter.textContent = fps;
    frameCount = 0;
    lastFpsUpdateTime = now;
  }
}

/**
 * ステータス表示の更新
 */
function updateStatus(msg) {
  statusText.textContent = msg;
}

/**
 * エラー表示
 */
function showError(msg) {
  errorBox.textContent = `ERROR: ${msg}`;
  errorBox.classList.remove("hidden");
}

// 解像度同期イベント
video.addEventListener("resize", updateCanvasResolution);
window.addEventListener("resize", updateCanvasResolution);

// DOM読み込み完了時に自動実行
window.addEventListener("DOMContentLoaded", init);

// ブラウザのAutoplay Policy対応（初回クリックまたはタップでAudioContextを確実にresume）
["pointerdown", "touchstart", "click", "keydown"].forEach((evt) => {
  window.addEventListener(evt, ensureAudioContext, { once: false, passive: true });
});

// TEST SOUNDボタン押下時の手動テスト発音
if (testSoundBtn) {
  testSoundBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    ensureAudioContext();
    playTapSound(587.33); // D5トーンでテスト発音
    console.log("[AUDIO] TEST SOUND 発音");
  });
}

// 手元CSVファイルの手動追加インプット (+FILE)
if (csvFileInput) {
  csvFileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const loadedList = [];
    for (const file of files) {
      const text = await file.text();
      loadedList.push({ name: file.name, content: text });
    }

    ingestCsvFiles(loadedList);
    csvFileInput.value = ""; // 連続選択可能にするためリセット
  });
}

// 初期ターゲット指（親指 THUMB）の設定
setTargetFinger("THUMB");

