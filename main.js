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
 * 初期化処理
 */
async function init() {
  resetDebugMetrics();
  updateStatus("モデル読込中...");

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

  // フロントカメラ（user）を確実に指定し、60fpsを最優先で要求
  const tryConstraints = [
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 60, min: 30 }
      },
      audio: false
    },
    {
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 60 }
      },
      audio: false
    },
    {
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 60 }
      },
      audio: false
    },
    {
      video: {
        facingMode: { ideal: "user" },
        frameRate: { ideal: 60 }
      },
      audio: false
    },
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
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

  // 実効カメラ設定（解像度・fps）をHUDに反映
  const track = stream.getVideoTracks()[0];
  if (track && typeof track.getSettings === "function") {
    const settings = track.getSettings();
    const fpsLabel = settings.frameRate ? `${Math.round(settings.frameRate)}fps` : "REQ 60fps";
    const resLabel = settings.height ? `${settings.height}p` : "";
    camInfo.textContent = `FRONT (${resLabel} ${fpsLabel})`.trim();
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
  requestAnimationFrame(predictLoop);

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
 * 毎フレームの推論ループ (requestAnimationFrame)
 */
function predictLoop() {
  if (!isPredicting) return;

  if (video.currentTime !== lastVideoTime && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    lastVideoTime = video.currentTime;

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

  requestAnimationFrame(predictLoop);
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
  const mcp = { x: landmarks[5].x * width, y: landmarks[5].y * height };
  const pip = { x: landmarks[6].x * width, y: landmarks[6].y * height };
  const dip = { x: landmarks[7].x * width, y: landmarks[7].y * height };
  const rawTip = { x: landmarks[8].x * width, y: landmarks[8].y * height };

  // 最新座標の保持
  currentTip.x = rawTip.x;
  currentTip.y = rawTip.y;

  // 3. デバッグHUDのリアルタイム表示更新（Raw座標）
  updateDebugMetrics(rawTip.x, rawTip.y);

  // 4. 人差し指の骨格描画（生検出座標ダイレクト追従・明確な白ソリッド線）
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 3;
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  canvasCtx.beginPath();
  canvasCtx.moveTo(mcp.x, mcp.y);
  canvasCtx.lineTo(pip.x, pip.y);
  canvasCtx.lineTo(dip.x, dip.y);
  canvasCtx.lineTo(rawTip.x, rawTip.y);
  canvasCtx.stroke();

  // 人差し指関節点（5: MCP, 6: PIP, 7: DIP）の描画（白丸＋黒枠）
  [mcp, pip, dip].forEach((pt) => {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fill();
    canvasCtx.strokeStyle = "#000000";
    canvasCtx.lineWidth = 1.5;
    canvasCtx.stroke();
  });

  // 5. 人差し指先端（TIP: Landmark 8）の白ターゲット丸マーク描画（生座標ダイレクト追従）
  drawTipTargetMark(rawTip.x, rawTip.y);
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
 * 人差し指TIPの生座標（X, Y）のHUD更新表示（白黒ミニマル）
 * @param {number} x
 * @param {number} y
 */
function updateDebugMetrics(x, y) {
  tipXVal.textContent = `${x.toFixed(1)}px`;
  tipYVal.textContent = `${y.toFixed(1)}px`;
}

/**
 * 手ロスト時のHUDメトリクスリセット
 */
function resetDebugMetrics() {
  tipXVal.textContent = "-";
  tipYVal.textContent = "-";
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
