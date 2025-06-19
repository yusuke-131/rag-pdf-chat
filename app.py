import streamlit as st
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma  
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI  
from langchain_ollama import OllamaLLM  
from langchain.chains import RetrievalQA  
from langchain.llms.base import LLM  
from huggingface_hub import InferenceClient  
from langchain import PromptTemplate  
import os
from dotenv import load_dotenv  
from datetime import datetime
import traceback
import uuid

# 環境変数読み込み
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # OpenAI APIキー
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") # HuggingFace Hub APIトークン


def get_hf_client():
    if not hf_token:
        raise ValueError("`.env` に HUGGINGFACEHUB_API_TOKEN が設定されていません。")
    return InferenceClient(token=hf_token)


class HFInferenceLLM(LLM):
    @property
    def client(self):
        return get_hf_client()

    def _call(self, prompt, stop=None):
        response = self.client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            inputs=prompt,
            parameters={"temperature": 0.7, "max_new_tokens": 256},
        )
        return response[0]["generated_text"]

    @property
    def _identifying_params(self):
        return {"model": "mistralai/Mistral-7B-Instruct-v0.1"}

    @property
    def _llm_type(self):
        return "huggingface_inference"


def main():
    st.title("📄 PDFベースのRAGチャットデモ")

    # サイドバー：LLMの選択
    llm_option = st.sidebar.radio(
        "使用するLLMを選択してください",
        ("OpenAI Chatモデル", "Ollama（ローカルモデル）"),
        index=0
    )

    # サイドバー：Embeddingの種類を選択
    embedding_option = st.sidebar.selectbox("Embeddingsの種類を選択してください", ["HuggingFace（無料）", "OpenAI（課金）"])

    # APIキーの存在チェックとエラーメッセージ表示
    if ("OpenAI" in llm_option or "OpenAI" in embedding_option) and not api_key:
        st.error("`.env` に OPENAI_API_KEY を設定してください。")
        st.stop()
    if llm_option == "HuggingFace Inference（無料）" and not hf_token:
        st.error("`.env` に HUGGINGFACEHUB_API_TOKEN を設定してください。")
        st.stop()

    # Embeddingsモデルの初期化
    if embedding_option == "HuggingFace（無料）":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # PDFアップロードUI
    uploaded_files = st.file_uploader("PDFをアップロードしてください", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        all_text = ""
        total_pages = 0
        st.subheader("アップロードされたPDFファイル")

        # アップロードされたPDFからテキスト抽出
        for uploaded_file in uploaded_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            total_pages += doc.page_count
            st.markdown(f"- **{uploaded_file.name}**（{doc.page_count}ページ）")
            for page in doc:
                all_text += page.get_text() + "\n"

        st.info(f"アップロードされたPDF数: {len(uploaded_files)}、合計ページ数: {total_pages}")

        # テキストをチャンクに分割（1000文字・100文字重複）
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_text(all_text)

        try:
            # Chromaベクトルストアの初期化
            collection_name = f"pdf_demo_{uuid.uuid4()}"
            vectordb = Chroma.from_texts(docs, embeddings, collection_name=collection_name)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            # LLMの選択に応じて初期化
            if llm_option == "OpenAI Chatモデル":
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
            elif llm_option == "Ollama（ローカルモデル）":
                llm = OllamaLLM(model="mistral", temperature=0.7)
            else:
                llm = HFInferenceLLM()
            
            # 日本語回答用のプロンプトテンプレート
            japanese_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""以下の文脈を参考にして質問に日本語で答えてください。
            文脈: {context}
            質問: {question}
            答え:"""
            )

            # RAG質問応答チェーンの初期化
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": japanese_prompt}
            )

        except Exception as e:
            st.error("埋め込みモデルまたはChromaの初期化中にエラーが発生しました。")
            st.code(traceback.format_exc())
            st.stop()

        # チャット履歴の初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.divider()
        st.subheader("💬 チャット")

        # 過去のチャットメッセージを表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f'`{message["time"]}`: {message["content"]}')

        # ユーザー入力を受け付け
        user_input = st.chat_input("質問を入力してください")
        if user_input:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.messages.append({"role": "user", "content": user_input, "time": timestamp})
            with st.chat_message("user"):
                st.markdown(f'`{timestamp}`: {user_input}')

            # AI回答を取得し表示
            with st.chat_message("assistant"):
                with st.spinner("考え中..."):
                    try:
                        answer = qa.invoke(user_input)
                        # 辞書で返ってきたら'result'だけ見やすく取り出す
                        if isinstance(answer, dict) and "result" in answer:
                            answer_text = answer["result"]
                        else:
                            answer_text = str(answer)
                    except Exception as e:
                        answer_text = f"エラーが発生しました：\n\n```\n{str(e)}\n```"
                    st.markdown(answer_text)

            st.session_state.messages.append({"role": "assistant", "content": answer_text, "time": timestamp})

        st.divider()

        # RAG構成図の表示（Mermaid形式）
        with st.expander("🧠 RAG構成図（Mermaid）"):
            embedding_label = "HuggingFace Embeddings" if "HuggingFace" in embedding_option else "OpenAI Embeddings"
            llm_label = {
                "OpenAI Chatモデル": "ChatGPT API",
                "Ollama（ローカルモデル）": "Ollama（ローカル）",
                "HuggingFace Inference（無料）": "HuggingFace Inference"
            }[llm_option]

            mermaid_code = f'''
graph TD
    A[PDFファイル群] --> B[テキスト抽出（PyMuPDF）]
    B --> C[テキスト分割（LangChain）]
    C --> D[ベクトル化（{embedding_label}）]
    D --> E[Chroma VectorStore]
    E --> F[Retriever（似た文を検索）]
    F --> G[{llm_label}]
    G --> H[StreamlitチャットUI]
'''
            st.code(mermaid_code, language="mermaid")

        # チャット履歴のテキストダウンロード機能
        with st.expander("⬇️ チャット履歴のダウンロード"):
            history_text = "\n\n".join(
                [f"[{m['time']}] {m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
            )
            st.download_button("テキスト形式で保存", history_text, file_name="chat_history.txt")


if __name__ == "__main__":
    main()