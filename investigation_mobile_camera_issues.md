# モバイル端末（OS・ブラウザ）におけるカメラ起動失敗の原因と対策

本ドキュメントでは、スマートフォン環境（iOS / Android、各種ブラウザアプリ、アプリ内ブラウザ）において、Webカメラ（`navigator.mediaDevices.getUserMedia`）が起動しない・映像が表示されない主な原因と、その解決策・実装上の対策についてまとめます。

---

## 1. 原因の全体像・分類

カメラが起動しないトラブルは、主に以下の6つのレイヤーに分類されます。

| 分類 | 主なエラー名・症状 | 発生しやすい環境 |
| :--- | :--- | :--- |
| **① セキュアコンテキスト制限** | `navigator.mediaDevices` が `undefined`<br>非セキュアコンテキストエラー | HTTP接続（ローカルIP `http://192.168.x.x` など） |
| **② アプリ内ブラウザ (WebView) 制限** | カメラが一切無反応、ブラックアウト<br>`NotAllowedError` | LINE、X（旧Twitter）、Instagram、TikTok など |
| **③ ユーザー操作ポリシー制限** | `NotAllowedError`<br>`video.play() failed` | ページ読み込み時の自動起動、低電力モード |
| **④ カメラ制約 (Constraints) エラー** | `OverconstrainedError` | 古い端末、フロントカメラのfps/解像度要求過多 |
| **⑤ パーミッション（権限）拒否・OS設定** | `NotAllowedError`<br>`PermissionDeniedError` | 初回に「拒否」を選択、OS設定でカメラ不許可 |
| **⑥ 他アプリによるカメラ占有** | `NotReadableError`<br>`TrackStartError` | カメラアプリや通話アプリがバックグラウンドで起動中 |

---

## 2. 各原因の詳細と技術的背景

### ① セキュアコンテキスト（HTTPS / SSL）の制限

#### 原因
ブラウザのセキュリティ仕様（W3C Media Capture and Streams 仕様）により、カメラやマイクなどのプライバシーに関わるハードウェアAPIは、**セキュアコンテキスト（HTTPS環境、または `localhost` / `127.0.0.1`）でのみ有効化**されます。

- **ローカルIP（`http://192.168.X.X:5173` 等）でアクセスした場合**:
  - PCブラウザの `localhost` では動作しますが、スマホから同一Wi-Fi内のローカルIPでアクセスすると **HTTP接続** となるため、ブラウザ側で `navigator.mediaDevices` 自体が `undefined` になるか、API呼出が即座にブロックされます。
- **自己署名証明書（オレオレ証明書）の警告**:
  - Viteのプラグイン等で自己署名HTTPSを導入している場合、iOS Safariや一部Androidブラウザで「安全ではない接続」警告を許可して進んでも、セキュリティ制約が解除されずカメラAPIが制限される場合があります。

#### 対策
- 開発時のスマホ実機検証には、**ngrok** や **Cloudflare Tunnel** などのトンネリングツールを利用して正式な有効証明書付きのHTTPS URLを発行する。
- ローカルCAツール（**mkcert** など）を使って自己署名ルート証明書をスマホ側にインストール・信頼設定する。
- 本番公開環境（Vercel, Netlify, GitHub Pages, Firebase Hosting など）のHTTPS上でテストする。

---

### ② アプリ内ブラウザ（In-App Browser / WebView）の機能制限

#### 原因
SNSアプリ（LINE、X/Twitter、Instagram、Facebook、TikTok）のトークやタイムライン上のリンクをタップした際に開く「アプリ内ブラウザ」は、OS標準のブラウザ（Safari / Chrome）とは異なり、WebView（アプリ内部のWeb表示コンポーネント）上で動作します。

- **iOS（WKWebView）の制約**:
  - iOS 14.3でようやく `getUserMedia` がサポートされましたが、親アプリ側がカメラ利用説明（`NSCameraUsageDescription`）を設定していない場合や、WebRTCのパーミッションハンドリング（`WKUIDelegate` の許可実装）を行っていない場合、カメラが起動しません。
  - 特に **LINEやXのiOS版アプリ内ブラウザ** では、カメラAPIの呼出が意図的に無効化・ブロックされているケースが非常に多いです。
- **Android（Android System WebView）の制約**:
  - 親アプリ側が `android.permission.CAMERA` パーミッションを保持していない場合、または `WebChromeClient.onPermissionRequest` を実装してWebからのリクエストを明示的に承認（grant）していない場合、カメラアクセスが自動的に拒否されます。

#### 対策
- **アプリ内ブラウザの検出と誘導**:
  User-Agent を判定し、アプリ内ブラウザである場合は「外部ブラウザ（Safari / Chrome）で開き直してください」と案内するモーダルを表示する。
  ```javascript
  const ua = navigator.userAgent || navigator.vendor || window.opera;
  const isLine = /Line\//i.test(ua);
  const isTwitter = /Twitter/i.test(ua);
  const isInstagram = /Instagram/i.test(ua);
  const isFB = /FB_IAB|FB4A|FBIOS/i.test(ua);

  if (isLine || isTwitter || isInstagram || isFB) {
    // 外部ブラウザへの誘導UIを表示
  }
  ```
- **LINEアプリ向けの外部ブラウザ強制パラメータ**:
  LINE内ブラウザで開かせず端末標準ブラウザで開かせるため、URLの末尾に `?openExternalBrowser=1` を付与する。

---

### ③ ユーザー操作ポリシー制限（自動起動・Autoplay制限）

#### 原因
現在の実装では、ページ読み込み直後にスクリプト（`init()`）から自動的に `startFrontCamera()` を呼出しています。

- **ユーザーアクション（User Gesture）の必須化**:
  近年のiOS SafariやモバイルChromeでは、プライバシー保護と意図しないバッテリー消費を防ぐため、**「ユーザーが画面を明示的にタップ・クリックしたイベント内」** でなければメディアデバイス（カメラ/マイク）の起動や、`<video>` タグの再生（`video.play()`）を拒否・保留するポリシーが強化されています。
- **低電力モード・省電力機能**:
  iPhoneの「低電力モード」がONになっている場合、自動再生ポリシーがさらに厳格化され、スクリプトからの自動的な `video.play()` が Promise reject されます。
- **`<video>` タグの必須属性**:
  iOS Safari でインライン（画面内）で動画を表示するには、`<video>` タグに **`playsinline`**（または `webkit-playsinline`）が必須です。これが欠けていると、OSの標準全画面メディアプレイヤーが立ち上がろうとしてエラーになります。

#### 対策
- ページロード時の自動起動に失敗した場合、画面中央に「**タップしてカメラを開始**」ボタンを表示し、ユーザーのタップをトリガーとして `startFrontCamera()` を再試行する設計にする。
- `<video>` タグには必ず以下の属性を付与する（現在実装済み）:
  ```html
  <video id="webcam" playsinline autoplay muted></video>
  ```
- `video.play()` の呼出は Promise のエラーを安全に catch する:
  ```javascript
  try {
    await video.play();
  } catch (err) {
    console.warn("video.play() 自動再生に失敗。ユーザー操作待機が必要です:", err);
  }
  ```

---

### ④ カメラ制約（MediaStreamConstraints）の過剰要求・非対応

#### 原因
現在 `startFrontCamera()` では、高品質なトラッキング（60fps・16:9比率）を目指して制約リスト（`tryConstraints`）を順に試行しています。

- **`min: 24` や `frameRate: 60` のハード制約**:
  Androidのローエンド・ミドルレンジ端末や、古いスマートフォンのインカメラは、ハードウェア的に **30fpsまでしか出力できない** 端末が多数存在します。`frameRate: { ideal: 60, min: 24 }` のように `min` を指定すると、ブラウザの実装によっては「要求条件を満たせない」と判定され、`OverconstrainedError` が発生して次のフォールバックに移行するまでにタイムアウトや不具合を起こすことがあります。
- **`track.getCapabilities()` / `track.applyConstraints()` の互換性**:
  iOS Safari では `MediaStreamTrack.getCapabilities()` が長らく非対応（未定義）であり、また `applyConstraints()` を呼ぶと例外を投げたりストリームがフリーズする既知の挙動があります。

#### 対策
- `tryConstraints` の段階的フォールバック設計を維持しつつ、最終フォールバックに最もシンプルな `{ video: { facingMode: "user" } }` または `{ video: true }` を確実に含める。
- `track.getCapabilities` や `track.applyConstraints` は、存在チェック（`typeof track.getCapabilities === "function"`）を行い、さらに `try-catch` でエラーを握りつぶしてストリームを破棄しないように保護する（現在の実装で一部対応済み）。

---

### ⑤ OS・ブラウザのパーミッション設定（拒否状態の記憶）

#### 原因
ユーザーが誤って一度でも「カメラのアクセスを許可しない（ブロック）」を選択した場合:

- **ブラウザの永続ブロック**:
  ブラウザはドメイン（オリジン）ごとに「拒否」の状態をキャッシュします。次回以降はユーザーに確認プロンプト（ダイアログ）を一切表示せず、即座に `NotAllowedError` を投げます。
- **OSレベルでのカメラ権限オフ**:
  - **iOS**: 「設定」アプリ >「Safari」>「カメラ」が「拒否」になっている場合。または「プライバシーとセキュリティ」>「カメラ」で Safari がオフになっている場合。
  - **Android**: 「設定」アプリ >「アプリ」>「Chrome」>「権限」>「カメラ」が「許可しない」になっている場合。
- **シークレットモード（プライベートブラウズ）**:
  セッションごとに権限がリセットされるか、ブラウザによってはカメラ利用自体が制限されている場合があります。

#### 対策
- `NotAllowedError` または `PermissionDeniedError` を検知した際、汎用的な「初期化に失敗しました」ではなく、**「ブラウザの設定でカメラがブロックされています。アドレスバーのアイコンまたは端末の設定からカメラを許可してください」** という具体的なリカバリー手順を画面上に明示する。

---

### ⑥ 他アプリによるカメラの排他占有（NotReadableError）

#### 原因
- スマートフォンのOSは、バッテリー節約やプライバシー保護のため、カメラセンサーを同時に1つのアプリしか占有できない仕様（排他的アクセス）が一般的です。
- ユーザーが直前まで「標準カメラアプリ」「QRコード読み取りアプリ」「LINE/Zoomなどの通話アプリ」を開いていた場合、それらのプロセスがカメラを掴んだままバックグラウンドに残っており、ブラウザ側で `NotReadableError` や `TrackStartError` が発生することがあります。

#### 対策
- `error.name === "NotReadableError"` をハンドリングし、「他のアプリ（カメラやビデオ通話など）がカメラを使用中です。他のアプリを完全に終了してから再度お試しください」と通知する。

---

## 3. 推奨される実装改善ロードマップ

| 項目 | 実装内容 | 期待効果 |
| :--- | :--- | :--- |
| **1. ユーザー操作契機の起動UI** | 自動起動に失敗した場合、画面に「カメラを起動する」ボタンを表示してタップで再実行 | 自動再生ポリシー・低電力モードでの起動失敗を防止 |
| **2. アプリ内ブラウザの検知・警告** | LINEやXなどのWebViewを検知し、標準ブラウザ（Safari / Chrome）で開くよう誘導 | WebView固有の起動不能問題を根本回避 |
| **3. エラー別の親切なガイダンス表示** | エラーオブジェクトの `name`（`NotAllowedError`, `NotReadableError`, `OverconstrainedError` 等）に応じた具体的な解決手順をHUD・トーストで表示 | ユーザーが自身で設定解除・解決できるよう支援 |
| **4. 制約の堅牢化** | `min` 制約などの過度な制約を外し、シンプルな制約へのフォールバックを高速化 | Android廉価端末や旧型端末での互換性向上 |
