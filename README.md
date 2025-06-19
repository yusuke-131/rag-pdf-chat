📄 PDFベースのRAGチャットデモ

このアプリは、アップロードされたPDFファイルを元に、RAG（Retrieval-Augmented Generation）の仕組みを使って質問に答えるチャットアプリです。
Streamlit で動作し、LangChain・Chroma・OpenAIやローカルLLM（Ollama）などを選択して利用できます。

💡 注意：このアプリは HuggingFace, OpenAI または Ollama（ローカルLLM）を使って動作します。
        そのため、.envの設定やモデルのダウンロードが必要です。

🔧 機能一覧

📎 複数PDFのアップロード

🧠 LangChainを用いたテキスト分割・ベクトル化

🔍 Chroma によるベクトル検索

💬 LLM（OpenAI / Ollama / HuggingFace）による回答生成

📥 チャット履歴のダウンロード（テキスト形式）

📊 MermaidによるRAG構成図の表示

💻 使用技術

分類

ライブラリ

Web UI

Streamlit

LLM

OpenAI / HuggingFace / Ollama

Embedding

sentence-transformers / OpenAI Embeddings

Vector Store

Chroma (chromadb)

テキスト分割

LangChain Text Splitter

PDF処理

PyMuPDF (fitz)

📦 インストールと起動

# リポジトリをクローン
git clone https://github.com/yusuke-131/rag-pdf-chat.git
cd rag-pdf-chat

# 仮想環境の作成（任意）
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 依存パッケージのインストール
pip install -r requirements.txt

# .env ファイルを作成し、以下を設定
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token

# アプリ起動
streamlit run app.py

※ Ollamaの利用には別途インストールとモデルのダウンロードが必要です。
  詳しくは [Ollama公式サイト](https://ollama.com/) をご覧ください。

🧠 Mermaidによる構成図

graph TD
    A[PDFファイル群] --> B[テキスト抽出（PyMuPDF）]
    B --> C[テキスト分割（LangChain）]
    C --> D[ベクトル化（Embeddings）]
    D --> E[Chroma VectorStore]
    E --> F[Retriever（類似文検索）]
    F --> G[LLM（OpenAI / Ollama / HF）]
    G --> H[StreamlitチャットUI]

📁 ファイル構成

├── app.py                # Streamlitメインアプリ
├── requirements.txt      # 必要なPythonパッケージ
└── .env                  # APIキーを保存（手動作成）

📝 今後の拡張案

チャット履歴の永続化（DBやファイル）

ファイルごとの検索フィルタ

マルチモーダル（画像＋PDF）対応

📄 ライセンス

MIT License