### ディレクトリ構成
```
Piarno_KBL_1D/
├── index.html              # メインUI（ビデオ/Canvas重畳、HUD、楽曲ガイド）
├── main.js                 # 推論・1 Euro Filter・打鍵判定・音響合成・楽曲管理
├── style.css               # 白黒基調のHUD & ネオンフィードバックUIスタイル
├── vite.config.js          # HTTPS対応およびLAN/QRコード公開設定
├── public/
│   ├── dataset/            # 指別の打鍵検証用CSVデータセット
│   │   ├── manifest.json   # 読み込み対象CSVの定義ファイル
│   │   └── *.csv           # 各指の時系列フレーム座標・打鍵ラベルデータ
│   └── models/             # MediaPipe HandLandmarker モデルアセット

```
### 必要環境

>依存関係のインストール
```bash
npm install
```
>開発サーバーの起動
```bash
npm run dev
```
詳細は [Markdownファイル](Recipe_of_Piarno.md) にまとめてあります。
