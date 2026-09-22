/**
 * 右手打鍵時系列CSVの左右反転スクリプト（Step 3: 命名是正・パース整合性検証・npm scripts統合）
 * 
 * public/dataset/ 配下の右手打鍵時系列CSVを再帰探索し、
 * 不自然なファイル名（空白・二重拡張子）を是正して左手用データセット（left_[finger]_[action]_[take].csv）を一括生成します。
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// 基準となるCanvas解像度幅（デフォルト 1280.0px）
export const DEFAULT_CANVAS_WIDTH = 1280.0;

/**
 * 出力先ファイル名の正規化・決定ルール
 * 不自然な空白の混入（例: "middle _"）や二重拡張子・ファイル結合（例: ".csvindex_..."）をクレンジングし、
 * "left_[finger]_[action]_[take].csv" 形式に正規化します。
 * @param {string} filename
 * @returns {string}
 */
export function generateLeftHandFileName(filename) {
  // 1. 空白を除去・アンダースコア前後の不要空白を整理
  let clean = filename.replace(/\s+/g, "");

  // 2. 途中に挟まった二重拡張子・連結部分（.csvindex_... や .csv...）を解消
  clean = clean.replace(/\.csv.+$/i, ".csv");

  // 3. 既存のプレフィックス（right_ や left_）をクリーンアップ
  clean = clean.replace(/^right_/, "").replace(/^left_/, "");

  // 4. index_tap01 を index_tap_01 のような標準命名に微調整
  clean = clean.replace(/_tap0(\d)/i, "_tap_0$1");

  return `left_${clean}`;
}

/**
 * 1行分のCSVレコードを反転処理
 * @param {string[]} cols カラム値の配列
 * @param {Record<string, number>} headerMap ヘッダー名からインデックスへのマップ
 * @param {number} canvasWidth 基準幅
 * @returns {{ newCols: string[], flipped: boolean, originalSample?: object, flippedSample?: object }}
 */
function flipCsvRow(cols, headerMap, canvasWidth) {
  const newCols = [...cols];
  const hasHandIdx = headerMap["has_hand"];

  // 手が検出されていない（has_hand === "0"）行はスキップして元の値を維持
  if (hasHandIdx !== undefined && hasHandIdx !== -1 && cols[hasHandIdx] !== undefined) {
    const hasHandVal = cols[hasHandIdx].trim();
    if (hasHandVal === "0" || hasHandVal === "") {
      return { newCols, flipped: false };
    }
  }

  let flippedAny = false;
  const originalSample = {};
  const flippedSample = {};

  // 1. ピクセル座標系（tip_x, mcp_x, wrist_x）の水平反転: W - x
  const targetXColumns = ["tip_x", "mcp_x", "wrist_x"];
  for (const colName of targetXColumns) {
    const idx = headerMap[colName];
    if (idx !== undefined && idx !== -1 && cols[idx] !== undefined) {
      const rawVal = cols[idx].trim();
      if (rawVal !== "") {
        const num = parseFloat(rawVal);
        if (!Number.isNaN(num)) {
          const flippedVal = (canvasWidth - num).toFixed(1);
          newCols[idx] = flippedVal;
          originalSample[colName] = rawVal;
          flippedSample[colName] = flippedVal;
          flippedAny = true;
        }
      }
    }
  }

  // 2. 先行データセット相対変位系（rx, vx）の符号反転: -rx, -vx
  const relativeXCols = ["rx", "vx"];
  for (const colName of relativeXCols) {
    const idx = headerMap[colName];
    if (idx !== undefined && idx !== -1 && cols[idx] !== undefined) {
      const rawVal = cols[idx].trim();
      if (rawVal !== "") {
        const num = parseFloat(rawVal);
        if (!Number.isNaN(num)) {
          const flippedVal = (-num).toString();
          newCols[idx] = flippedVal;
          originalSample[colName] = rawVal;
          flippedSample[colName] = flippedVal;
          flippedAny = true;
        }
      }
    }
  }

  return {
    newCols,
    flipped: flippedAny,
    originalSample: flippedAny ? originalSample : undefined,
    flippedSample: flippedAny ? flippedSample : undefined
  };
}

/**
 * CSV文字列全体の左右反転処理
 * @param {string} csvText 入力CSVテキスト
 * @param {number} canvasWidth 基準幅 (デフォルト 1280.0)
 * @returns {{ flippedText: string, totalRows: number, flippedRows: number, samples: Array<{ row: number, before: object, after: object }> }}
 */
export function flipCsvContent(csvText, canvasWidth = DEFAULT_CANVAS_WIDTH) {
  if (!csvText || typeof csvText !== "string") {
    throw new Error("入力CSVテキストが空または無効です。");
  }

  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error("CSV行数が不足しています（ヘッダーと最低1行のデータが必要です）。");
  }

  // ヘッダー解析
  const headerLine = lines[0];
  const rawHeaders = headerLine.split(",").map((h) => h.trim());
  const headerMap = {};
  rawHeaders.forEach((h, idx) => {
    headerMap[h.toLowerCase()] = idx;
  });

  const outputLines = [headerLine];
  let totalRows = 0;
  let flippedRows = 0;
  const samples = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    totalRows++;
    const cols = line.split(",").map((c) => c.trim());
    const res = flipCsvRow(cols, headerMap, canvasWidth);

    if (res.flipped) {
      flippedRows++;
      if (samples.length < 3 || i === lines.length - 2) {
        samples.push({
          row: i,
          before: res.originalSample,
          after: res.flippedSample
        });
      }
    }

    outputLines.push(res.newCols.join(","));
  }

  return {
    flippedText: outputLines.join("\n") + "\n",
    totalRows,
    flippedRows,
    samples
  };
}

/**
 * 単一CSVファイルの左右反転と出力
 * @param {string} inputPath 変換元CSVファイルパス
 * @param {string} outputPath 出力先CSVファイルパス
 * @param {object} options オプション（canvasWidth, overwrite）
 * @returns {{ status: "SUCCESS" | "SKIPPED" | "ERROR", inputPath: string, outputPath: string, message?: string }}
 */
export function flipCsvFile(inputPath, outputPath, options = {}) {
  const { canvasWidth = DEFAULT_CANVAS_WIDTH, overwrite = false } = options;
  const resolvedInput = path.resolve(inputPath);
  const resolvedOutput = path.resolve(outputPath);

  if (!fs.existsSync(resolvedInput)) {
    throw new Error(`入力ファイルが存在しません: ${resolvedInput}`);
  }

  // 上書き防止ガード
  if (!overwrite && fs.existsSync(resolvedOutput)) {
    return {
      status: "SKIPPED",
      inputPath: resolvedInput,
      outputPath: resolvedOutput,
      message: "出力先ファイルが既に存在するためスキップしました（上書き防止）"
    };
  }

  const csvContent = fs.readFileSync(resolvedInput, "utf-8");
  const result = flipCsvContent(csvContent, canvasWidth);

  // 出力先ディレクトリの自動作成
  const outputDir = path.dirname(resolvedOutput);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(resolvedOutput, result.flippedText, "utf-8");

  return {
    status: "SUCCESS",
    inputPath: resolvedInput,
    outputPath: resolvedOutput,
    totalRows: result.totalRows,
    flippedRows: result.flippedRows,
    samples: result.samples
  };
}

/**
 * ディレクトリ内のCSVファイルを再帰的に探索
 * @param {string} dir
 * @returns {string[]} CSVファイルパスの配列
 */
function findCsvFilesRecursively(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findCsvFilesRecursively(fullPath));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) {
      results.push(fullPath);
    }
  }
  return results;
}

/**
 * 指定ディレクトリ配下のCSVファイルを一括反転バッチ処理
 * @param {string} targetDir 対象ディレクトリ（例: public/dataset）
 * @param {object} options
 */
export function flipDatasetBatch(targetDir = "public/dataset", options = {}) {
  const resolvedDir = path.resolve(targetDir);
  console.log(`\n======================================================`);
  console.log(`[BATCH] データセット一括反転バッチ処理を開始 (Step 3)`);
  console.log(`[BATCH] 探索ディレクトリ: ${resolvedDir}`);
  console.log(`======================================================\n`);

  if (!fs.existsSync(resolvedDir)) {
    throw new Error(`対象ディレクトリが存在しません: ${resolvedDir}`);
  }

  const allCsvFiles = findCsvFilesRecursively(resolvedDir);
  console.log(`[BATCH] 検出された総CSVファイル数: ${allCsvFiles.length} 件`);

  // 対象ファイルのフィルタリング（left_で始まるファイル、一時検証ファイルは除外）
  const candidateFiles = allCsvFiles.filter((filePath) => {
    const fileName = path.basename(filePath);
    if (fileName.startsWith("left_")) return false; // 既に反転済みのファイルは除外
    if (fileName === "test_left_output.csv") return false;
    return true;
  });

  console.log(`[BATCH] 反転変換対象ファイル数: ${candidateFiles.length} 件\n`);

  const successList = [];
  const skippedList = [];
  const errorList = [];

  for (const inputPath of candidateFiles) {
    const dirName = path.dirname(inputPath);
    const fileName = path.basename(inputPath);
    const newFileName = generateLeftHandFileName(fileName);
    const outputPath = path.join(dirName, newFileName);

    try {
      const res = flipCsvFile(inputPath, outputPath, options);
      if (res.status === "SUCCESS") {
        successList.push({ input: inputPath, output: outputPath, flippedRows: res.flippedRows });
        console.log(`[SUCCESS] ${fileName} -> ${newFileName} (${res.flippedRows}行反転)`);
      } else if (res.status === "SKIPPED") {
        skippedList.push({ input: inputPath, output: outputPath, reason: res.message });
        console.log(`[SKIPPED] ${fileName} -> ${newFileName} (${res.message})`);
      }
    } catch (err) {
      errorList.push({ input: inputPath, error: err.message });
      console.error(`[ERROR] ${fileName} の変換失敗:`, err.message);
    }
  }

  // 実行後サマリーの表示
  console.log(`\n================== [実行結果サマリー] ==================`);
  console.log(`  変換成功: ${successList.length} 件`);
  console.log(`  スキップ: ${skippedList.length} 件`);
  console.log(`  エラー  : ${errorList.length} 件`);
  console.log(`======================================================\n`);

  if (successList.length > 0) {
    console.log(`[生成された正規化左手ファイル一覧 (${successList.length} 件)]:`);
    successList.forEach((item, idx) => {
      const relOutput = path.relative(process.cwd(), item.output);
      console.log(`  ${idx + 1}. ${relOutput}`);
    });
    console.log("");
  }

  return {
    totalCandidates: candidateFiles.length,
    successCount: successList.length,
    skippedCount: skippedList.length,
    errorCount: errorList.length,
    successList,
    skippedList,
    errorList
  };
}

/**
 * main.js の parseCsv 互換パーサーによるパース整合性検証
 * @param {string} filePath
 * @returns {{ valid: boolean, totalFrames: number, hitFrames: number, minX: number, maxX: number, issues: string[] }}
 */
export function verifyCsvIntegrity(filePath) {
  const text = fs.readFileSync(filePath, "utf-8");
  const lines = text.trim().split(/\r?\n/);
  const issues = [];

  if (lines.length < 2) {
    issues.push("行数が不足しています");
    return { valid: false, totalFrames: 0, hitFrames: 0, minX: NaN, maxX: NaN, issues };
  }

  const headerLine = lines[0].toLowerCase();
  const headers = headerLine.split(",").map((h) => h.trim());

  const ryIdx = headers.indexOf("ry");
  const labelIdx = headers.indexOf("label") !== -1 ? headers.indexOf("label") : headers.indexOf("is_hit");
  const timeIdx = headers.indexOf("timestamp") !== -1 ? headers.indexOf("timestamp") : (headers.indexOf("timestamp_ms") !== -1 ? headers.indexOf("timestamp_ms") : headers.indexOf("time_sec"));
  const tipXIdx = headers.indexOf("tip_x") !== -1 ? headers.indexOf("tip_x") : headers.indexOf("index_tip_x");
  const rxIdx = headers.indexOf("rx");

  let validCount = 0;
  let hitCount = 0;
  let minX = Infinity;
  let maxX = -Infinity;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(",").map((c) => c.trim());

    // X座標のチェック
    if (tipXIdx !== -1 && cols[tipXIdx]) {
      const xVal = parseFloat(cols[tipXIdx]);
      if (Number.isNaN(xVal)) {
        issues.push(`行 ${i}: tip_x が NaN です (${cols[tipXIdx]})`);
      } else {
        if (xVal < minX) minX = xVal;
        if (xVal > maxX) maxX = xVal;
      }
    } else if (rxIdx !== -1 && cols[rxIdx]) {
      const rxVal = parseFloat(cols[rxIdx]);
      if (Number.isNaN(rxVal)) {
        issues.push(`行 ${i}: rx が NaN です (${cols[rxIdx]})`);
      } else {
        if (rxVal < minX) minX = rxVal;
        if (rxVal > maxX) maxX = rxVal;
      }
    }

    // ryのチェック
    if (ryIdx !== -1 && cols[ryIdx]) {
      const ryVal = parseFloat(cols[ryIdx]);
      if (Number.isNaN(ryVal)) {
        issues.push(`行 ${i}: ry が NaN です (${cols[ryIdx]})`);
      }
    }

    // 打鍵フラグ
    if (labelIdx !== -1 && cols[labelIdx] === "1") {
      hitCount++;
    }

    validCount++;
  }

  return {
    valid: issues.length === 0,
    totalFrames: validCount,
    hitFrames: hitCount,
    minX: minX === Infinity ? 0 : minX,
    maxX: maxX === -Infinity ? 0 : maxX,
    issues
  };
}

// CLIから直接実行された場合のハンドラ
const isMainModule = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (isMainModule) {
  const args = process.argv.slice(2);
  const isForce = args.includes("--force") || args.includes("--overwrite");
  const targetDir = args.find((a) => !a.startsWith("--")) || "public/dataset";

  try {
    const res = flipDatasetBatch(targetDir, { overwrite: isForce });

    // Step 3 整合性自動検証（生成された左手CSVファイルを全件スキャン）
    const leftCsvs = findCsvFilesRecursively(path.resolve(targetDir)).filter((f) =>
      path.basename(f).startsWith("left_")
    );

    console.log(`================== [パース整合性検証] ==================`);
    console.log(`対象左手CSVファイル数: ${leftCsvs.length} 件`);

    let allValid = true;
    for (const leftFile of leftCsvs) {
      const vRes = verifyCsvIntegrity(leftFile);
      const relPath = path.relative(process.cwd(), leftFile);
      if (vRes.valid) {
        console.log(`[PASS] ${relPath} (${vRes.totalFrames}行, 打鍵: ${vRes.hitFrames}件, X範囲: ${vRes.minX.toFixed(1)}〜${vRes.maxX.toFixed(1)})`);
      } else {
        allValid = false;
        console.error(`[FAIL] ${relPath}: ${vRes.issues.join(", ")}`);
      }
    }
    console.log(`======================================================`);
    if (allValid) {
      console.log(`[SUCCESS] 全ての左手CSVファイルでパース整合性・座標数値が正常です！\n`);
    } else {
      console.error(`[WARN] 一部のファイルでパース検証エラーが発生しました。\n`);
    }
  } catch (err) {
    console.error(`[FATAL] バッチ処理が中断しました:`, err.message);
    process.exit(1);
  }
}
