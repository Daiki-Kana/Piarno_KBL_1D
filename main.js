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
    audioCtx.resume();
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

  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  // トライアングル波（ピアノに似た倍音成分）
  osc.type = "triangle";
  osc.frequency.setValueAtTime(freq, now);

  // エンベロープ（即座に立ち上がり、約0.15秒で減衰）
  gain.gain.setValueAtTime(0.001, now);
  gain.gain.linearRampToValueAtTime(0.35, now + 0.003);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(now + 0.16);

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

// 人差し指4関節（MCP, PIP, DIP, TIP）専用の適応フィルター
const indexPointFilters = {
  mcp: new PointFilter(1.5, 0.01),
  pip: new PointFilter(1.5, 0.01),
  dip: new PointFilter(1.5, 0.01),
  tip: new PointFilter(1.5, 0.01)
};

// 手の骨格コネクション定義（全21ランドマーク間の接続）
const HAND_CONNECTIONS = [
  // 手のひら
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  // 親指
  [1, 2], [2, 3], [3, 4],
  // 人差し指
  [5, 6], [6, 7], [7, 8],
  // 中指
  [9, 10], [10, 11], [11, 12],
  // 薬指
  [13, 14], [14, 15], [15, 16],
  // 小指
  [17, 18], [18, 19], [19, 20]
];

// 人差し指のランドマークインデックス (5: MCP, 6: PIP, 7: DIP, 8: TIP)
const INDEX_LANDMARK_INDICES = [5, 6, 7, 8];

// 人差し指以外の骨格接続（背景薄表示用）
const OTHER_CONNECTIONS = HAND_CONNECTIONS.filter(
  ([s, e]) => !(INDEX_LANDMARK_INDICES.includes(s) && INDEX_LANDMARK_INDICES.includes(e))
);

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
export function parseCsv(text) {
  if (!text || typeof text !== "string") return [];
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const headerLine = lines[0].toLowerCase();
  const headers = headerLine.split(",").map((h) => h.trim());

  const ryIdx = headers.indexOf("ry");
  const labelIdx = headers.indexOf("label");
  const timeIdx = headers.indexOf("timestamp") !== -1 ? headers.indexOf("timestamp") : headers.indexOf("timestamp_ms");

  const mcpYIdx = headers.indexOf("index_mcp_y");
  const tipYIdx = headers.indexOf("index_tip_y");
  const mcpXIdx = headers.indexOf("index_mcp_x");
  const tipXIdx = headers.indexOf("index_tip_x");

  const results = [];

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
    }

    if (ryIdx !== -1 && cols[ryIdx] !== undefined) {
      // timestamp,rx,ry,rz... 形式
      ry = parseFloat(cols[ryIdx]);
    } else if (mcpYIdx !== -1 && tipYIdx !== -1 && cols[mcpYIdx] && cols[tipYIdx]) {
      // 関節座標形式: ry = (tip.y - mcp.y)
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

  return results;
}

/**
 * 複数CSVテキストを一括パースしてデータセット配列に格納
 * @param {Array<{ name: string, content: string }>} files
 */
function ingestCsvFiles(files) {
  for (const file of files) {
    const parsed = parseCsv(file.content);
    if (parsed.length > 0) {
      // 既存の同名ファイルがあれば差し替え、なければ追加
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
      `[CSV] 全読込成功: 計 ${testDataDatasets.length} ファイル, ${testDataFrames.length} フレーム (打鍵マーク: ${totalHits})`,
      testDataDatasets.map((d) => `${d.filename} (${d.frames.length}f)`)
    );
  } else {
    updateCsvStatus("NOT LOADED");
    if (thVal) thVal.textContent = hitRyThreshold.toFixed(2);
  }
}

/**
 * 読み込んだ学習済みCSVデータから打鍵判定閾値を自動導出
 * （空中動作などの非打鍵データの上限値を確実に超える安全マージンを設定）
 */
function trainModelFromCsv() {
  // 1. 全フレームから label: 1 (打鍵) の ry 値を抽出
  const hitRyValues = testDataFrames
    .filter((f) => f.isHit && f.ry !== null && !isNaN(f.ry))
    .map((f) => f.ry)
    .sort((a, b) => a - b);

  // 2. 空中動作データ (index_neg_air) または非打鍵フレーム (label === 0) の最大 ry 値を探索
  const airDataset = testDataDatasets.find((d) => d.filename.toLowerCase().includes("neg_air"));
  let maxAirRy = -Infinity;
  if (airDataset) {
    for (const f of airDataset.frames) {
      if (f.ry !== null && !isNaN(f.ry)) {
        if (f.ry > maxAirRy) maxAirRy = f.ry;
      }
    }
  }

  // 空中データファイルが明示的にない場合は全非打鍵データの上位90%点を参考
  if (maxAirRy === -Infinity) {
    const negRys = testDataFrames
      .filter((f) => !f.isHit && f.ry !== null && !isNaN(f.ry))
      .map((f) => f.ry)
      .sort((a, b) => a - b);
    if (negRys.length > 0) {
      maxAirRy = negRys[Math.floor(negRys.length * 0.90)];
    } else {
      maxAirRy = 0.72; // デフォルト空中最大変位
    }
  }

  trainedHitSamples = hitRyValues.length;

  if (trainedHitSamples > 0) {
    // 打鍵サンプルの中央値（p50）および下位25%点（p25）
    const p25 = hitRyValues[Math.floor(trainedHitSamples * 0.25)];
    const p50 = hitRyValues[Math.floor(trainedHitSamples * 0.50)];

    // 空中最大変位（約0.714）に安全マージン（+0.08）を加えた値と、打鍵サンプルの適正範囲から閾値を算出
    // （空中誤検知を確実に遮断するため最低でも0.80以上に設定）
    const airSafeTh = maxAirRy + 0.08;
    hitRyThreshold = Math.max(0.80, Math.min(p50, Math.max(airSafeTh, p25)));

    // リフト復帰閾値（指を打鍵位置からしっかり持ち上げるまで再打鍵を遮断）
    liftRyThreshold = Math.min(hitRyThreshold * 0.68, maxAirRy * 0.85);

    console.log(
      `[MODEL] 学習完了: 打鍵サンプル ${trainedHitSamples} 件 (空中Max=${maxAirRy.toFixed(3)}) -> 打鍵TH=${hitRyThreshold.toFixed(2)}, リフトTH=${liftRyThreshold.toFixed(2)}`
    );
  } else {
    hitRyThreshold = 0.82;
    liftRyThreshold = 0.55;
  }

  if (thVal) {
    thVal.textContent = hitRyThreshold.toFixed(2);
  }
}

/**
 * public/dataset/ 配下のCSVデータセットを自動非同期フェッチして読み込む
 */
export async function loadTestDataCsv() {
  const loadedFiles = [];

  // 1. public/dataset/manifest.json からファイル一覧を取得して非同期fetch
  try {
    const manifestRes = await fetch("./dataset/manifest.json");
    if (manifestRes.ok) {
      const fileList = await manifestRes.json();
      if (Array.isArray(fileList)) {
        for (const fname of fileList) {
          try {
            const csvRes = await fetch(`./dataset/${fname}`);
            if (csvRes.ok) {
              const text = await csvRes.text();
              loadedFiles.push({ name: fname, content: text });
            }
          } catch (err) {
            console.warn(`[CSV] ${fname} の取得スキップ:`, err);
          }
        }
      }
    }
  } catch (err) {
    console.warn("[CSV] manifest.json の読み込みをスキップ:", err);
  }

  // 取得できたCSVファイル群をデータセットへ登録
  if (loadedFiles.length > 0) {
    ingestCsvFiles(loadedFiles);
  } else {
    updateCsvStatus("NO CSV DATA");
    if (thVal) thVal.textContent = hitRyThreshold.toFixed(2);
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

  // 60fpsを最優先で要求（720pおよび480p/360pでの60fpsモードを優先探索）
  const tryConstraints = [
    // 1. 720p + 60fps厳格指定 (min: 55 または exact: 60)
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280, max: 1280 },
        height: { ideal: 720, max: 720 },
        frameRate: { ideal: 60, min: 55 }
      },
      audio: false
    },
    {
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 1280, max: 1280 },
        height: { ideal: 720, max: 720 },
        frameRate: { exact: 60 }
      },
      audio: false
    },
    // 2. 480p + 60fps（720pで60fps非対応のカメラでも480pなら60fpsを出せる機種向け）
    {
      video: {
        facingMode: "user",
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        frameRate: { ideal: 60, min: 55 }
      },
      audio: false
    },
    {
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        frameRate: { exact: 60 }
      },
      audio: false
    },
    // 3. 360p + 60fps
    {
      video: {
        facingMode: "user",
        width: { ideal: 640, max: 640 },
        height: { ideal: 360, max: 360 },
        frameRate: { ideal: 60, min: 50 }
      },
      audio: false
    },
    // 4. 任意解像度（最大720p）+ 60fps
    {
      video: {
        facingMode: { ideal: "user" },
        width: { max: 1280 },
        height: { max: 720 },
        frameRate: { ideal: 60, min: 50 }
      },
      audio: false
    },
    // 5. フォールバック
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280, max: 1280 },
        height: { ideal: 720, max: 720 },
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

  // ハードウェアがサポートする最大フレームレート（最大60fps）を強制適用
  const track = stream.getVideoTracks()[0];
  if (track) {
    if (typeof track.getCapabilities === "function") {
      const caps = track.getCapabilities();
      if (caps.frameRate && caps.frameRate.max >= 50) {
        try {
          const targetFps = Math.min(60, caps.frameRate.max);
          await track.applyConstraints({
            frameRate: { ideal: targetFps, min: Math.min(55, targetFps) }
          });
        } catch (fpsErr) {
          console.warn("最大fps制約の適用スキップ:", fpsErr);
        }
      }
    }

    // 実効カメラ設定（解像度・fps）をHUDに反映（720p超過時は上限適用を試行）
    if (typeof track.getSettings === "function") {
      let settings = track.getSettings();
      if (settings.height && settings.height > 720) {
        try {
          await track.applyConstraints({
            width: { ideal: 1280, max: 1280 },
            height: { ideal: 720, max: 720 }
          });
          settings = track.getSettings();
        } catch (err) {
          console.warn("720p上限の追加適用をスキップ:", err);
        }
      }
      const fpsLabel = settings.frameRate ? `${Math.round(settings.frameRate)}fps` : "REQ 60fps";
      const resLabel = settings.height ? `${settings.height}p` : "";
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

  if (!results || !results.landmarks || results.landmarks.length === 0) {
    handsCount.textContent = "0";
    if (tapState !== "IDLE") {
      tapState = "IDLE";
      updateStateHud("IDLE", false);
    }
    // 手ロスト時にフィルターをリセット（再検出時の座標ジャンプ防止）
    indexPointFilters.mcp.reset();
    indexPointFilters.pip.reset();
    indexPointFilters.dip.reset();
    indexPointFilters.tip.reset();
    resetDebugMetrics();
    return;
  }

  handsCount.textContent = `${results.landmarks.length}`;

  const width = canvas.width;
  const height = canvas.height;

  // メインの手（先頭の手）を対象に人差し指を描画
  const landmarks = results.landmarks[0];

  // 1. 他の指・手のひら（非常に薄いグレースケールで描画）
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  canvasCtx.lineWidth = 1;
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  OTHER_CONNECTIONS.forEach(([startIdx, endIdx]) => {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];

    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * width, p1.y * height);
    canvasCtx.lineTo(p2.x * width, p2.y * height);
    canvasCtx.stroke();
  });

  // 人差し指以外の関節点（極小薄グレー）
  landmarks.forEach((p, idx) => {
    if (INDEX_LANDMARK_INDICES.includes(idx)) return;
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

  // 2. 人差し指ランドマークの生ピクセル座標 (5: MCP, 6: PIP, 7: DIP, 8: TIP)
  const rawMcp = { x: landmarks[5].x * width, y: landmarks[5].y * height };
  const rawPip = { x: landmarks[6].x * width, y: landmarks[6].y * height };
  const rawDip = { x: landmarks[7].x * width, y: landmarks[7].y * height };
  const rawTip = { x: landmarks[8].x * width, y: landmarks[8].y * height };

  // 1 Euro Filterによる適応平滑化（静止時はジッター完全除去、打鍵時は遅延ゼロ追従）
  const now = performance.now();
  const smoothMcp = indexPointFilters.mcp.filter(rawMcp.x, rawMcp.y, now);
  const smoothPip = indexPointFilters.pip.filter(rawPip.x, rawPip.y, now);
  const smoothDip = indexPointFilters.dip.filter(rawDip.x, rawDip.y, now);
  const smoothTip = indexPointFilters.tip.filter(rawTip.x, rawTip.y, now);

  // 最新平滑化座標の保持
  currentTip.x = smoothTip.x;
  currentTip.y = smoothTip.y;

  // 手首（Landmark 0: Wrist）と人差し指付け根（Landmark 5: MCP）間の3D距離 L（手の基準長）
  const dx05 = landmarks[5].x - landmarks[0].x;
  const dy05 = landmarks[5].y - landmarks[0].y;
  const dz05 = (landmarks[5].z ?? 0) - (landmarks[0].z ?? 0);
  const L = Math.hypot(dx05, dy05, dz05) || 0.001;

  // 平滑化座標に基づく安定した相対変位 ry（ジッターによる微小スパイクを排除）
  const currentRy = (smoothTip.y - smoothMcp.y) / (height * L);
  const rawRy = (landmarks[8].y - landmarks[5].y) / L;

  // 3. 学習済みCSVモデルによるリアルタイム打鍵認識（空中誤検知を遮断）
  if (tapState === "IDLE") {
    // 平滑化変位と生変位の両方が安全閾値以上の場合のみ打鍵判定
    if (currentRy >= hitRyThreshold && rawRy >= hitRyThreshold * 0.95) {
      tapState = "TOUCHED";
      playTapSound(523.25); // C5打鍵音
      updateStateHud("TOUCHED", true);
      console.log(`[REALTIME TAP] #${tapCount} ry=${currentRy.toFixed(3)} (Raw=${rawRy.toFixed(3)}) >= TH:${hitRyThreshold.toFixed(2)}`);
    }
  } else if (tapState === "TOUCHED") {
    if (currentRy < liftRyThreshold) {
      // 指の持ち上がり（リフト）を検知して待機状態に復帰
      tapState = "IDLE";
      updateStateHud("IDLE", false);
    }
  }

  // 4. デバッグHUDのリアルタイム表示更新（平滑化座標と相対変位）
  updateDebugMetrics(smoothTip.x, smoothTip.y, currentRy);

  // 5. 人差し指の骨格描画（ブレのない平滑化ライン）
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 3;
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  canvasCtx.beginPath();
  canvasCtx.moveTo(smoothMcp.x, smoothMcp.y);
  canvasCtx.lineTo(smoothPip.x, smoothPip.y);
  canvasCtx.lineTo(smoothDip.x, smoothDip.y);
  canvasCtx.lineTo(smoothTip.x, smoothTip.y);
  canvasCtx.stroke();

  // 人差し指関節点（5: MCP, 6: PIP, 7: DIP）の描画（白丸＋黒枠）
  [smoothMcp, smoothPip, smoothDip].forEach((pt) => {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fill();
    canvasCtx.strokeStyle = "#000000";
    canvasCtx.lineWidth = 1.5;
    canvasCtx.stroke();
  });

  // 6. 人差し指先端（TIP: Landmark 8）の白ターゲット丸マーク描画（ジッターゼロ）
  drawTipTargetMark(smoothTip.x, smoothTip.y);
}

/**
 * 人差し指先端（TIP）のターゲット丸マーク描画（白黒ミニマル・ダイレクト追従）
 * @param {number} x
 * @param {number} y
 */
function drawTipTargetMark(x, y) {
  // 外側のターゲットサークル
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 9, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();

  // 十字ターゲットインジケータ
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 1.5;
  canvasCtx.beginPath();
  // 水平線
  canvasCtx.moveTo(x - 13, y);
  canvasCtx.lineTo(x - 5, y);
  canvasCtx.moveTo(x + 5, y);
  canvasCtx.lineTo(x + 13, y);
  // 垂直線
  canvasCtx.moveTo(x, y - 13);
  canvasCtx.lineTo(x, y - 5);
  canvasCtx.moveTo(x, y + 5);
  canvasCtx.lineTo(x, y + 13);
  canvasCtx.stroke();

  // 中心ターゲット丸
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 3.5, 0, 2 * Math.PI);
  canvasCtx.fillStyle = "#ffffff";
  canvasCtx.fill();
  canvasCtx.strokeStyle = "#000000";
  canvasCtx.lineWidth = 1;
  canvasCtx.stroke();
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

// ブラウザのAutoplay Policy対応（初回クリックまたはタップでAudioContextをresume）
window.addEventListener("pointerdown", ensureAudioContext, { once: false });
window.addEventListener("keydown", ensureAudioContext, { once: false });

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

