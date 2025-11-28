import os
import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# .env 読み込み & 環境変数
# ---------------------------------------------------------
load_dotenv()

# チャット用 LLM (Locallm 側)
LOCALLM_API_KEY = os.getenv("LOCALLM_API_KEY")
LOCALLM_BASE_URL = os.getenv("LOCALLM_BASE_URL")
LOCALLM_CHAT_MODEL = os.getenv("LOCALLM_CHAT_MODEL")

# 埋め込み用モデル
LOCALLM_EMBEDDING_MODEL = os.getenv("LOCALLM_EMBEDDING_MODEL")

# 埋め込み専用エンドポイント（任意）
EMB_API_KEY = os.getenv("EMB_API_KEY") or LOCALLM_API_KEY
EMB_BASE_URL = os.getenv("EMB_BASE_URL") or LOCALLM_BASE_URL

# ---------------------------------------------------------
# パス設定
# app_emb.py は app/ 配下にある想定
# ルート:
#   Locallm/
#     app/app_emb.py
#     data/knowledge.txt
#     data/system_prompt.txt
#     logs/
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# セッション ID を日付 + ランダムで生成
def get_session_id() -> str:
    if "session_id" not in st.session_state:
        date_str = datetime.now().strftime("%Y%m%d")
        rand = uuid.uuid4().hex[:8]
        st.session_state.session_id = f"{date_str}_{rand}"
    return st.session_state.session_id


# ---------------------------------------------------------
# ログ書き込み（1行1JSON の jsonl 形式）
#   ファイル名: logs/<session_id>.jsonl
# ---------------------------------------------------------
def log_interaction(
    question: str,
    answer: str,
    contexts: List[str],
    extra: Dict[str, Any] | None = None,
) -> None:
    """logs/<session_id>.jsonl に Q&A とコンテキストを追記"""
    extra = extra or {}
    session_id = get_session_id()
    log_path = LOGS_DIR / f"{session_id}.jsonl"

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    record.update(extra)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_log_files() -> List[Path]:
    """logs/ 配下の *.jsonl を新しい順に返す"""
    files = sorted(LOGS_DIR.glob("*.jsonl"), reverse=True)
    return files


def load_history_from_log(log_path: Path) -> List[Dict[str, str]]:
    """logs/<session>.jsonl から history を組み立てる"""
    history: List[Dict[str, str]] = []
    if not log_path.exists():
        return history

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = rec.get("question")
            a = rec.get("answer")
            if q and a:
                history.append({"user": q, "assistant": a})
    return history


# ---------------------------------------------------------
# クライアント生成
# ---------------------------------------------------------
def get_chat_client():
    """チャット用 LLM クライアント"""
    if not LOCALLM_API_KEY:
        return "LOCALLM_API_KEY が設定されていません。.env を確認してください。"
    if not LOCALLM_BASE_URL:
        return "LOCALLM_BASE_URL が設定されていません。.env を確認してください。"
    if not LOCALLM_CHAT_MODEL:
        return "LOCALLM_CHAT_MODEL が設定されていません。.env を確認してください。"

    client = OpenAI(
        api_key=LOCALLM_API_KEY,
        base_url=LOCALLM_BASE_URL,
    )
    return client


def get_embedding_client():
    """埋め込み用クライアント

    - EMB_API_KEY / EMB_BASE_URL があればそちらを優先
    - なければ LOCALLM_* を利用（同じエンドポイントで埋め込みを取る）
    """
    if not EMB_API_KEY:
        return "EMB_API_KEY もしくは LOCALLM_API_KEY が設定されていません。.env を確認してください。"

    # EMB_BASE_URL が空で LOCALLM_BASE_URL も空な場合は OpenAI デフォルトに倒す
    base_url = EMB_BASE_URL or "https://api.openai.com/v1"

    if not LOCALLM_EMBEDDING_MODEL:
        return "LOCALLM_EMBEDDING_MODEL が設定されていません（埋め込みモデル名）。.env を確認してください。"

    client = OpenAI(
        api_key=EMB_API_KEY,
        base_url=base_url,
    )
    return client


# ---------------------------------------------------------
# system_prompt.txt 読み込み
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    """
    data/system_prompt.txt の内容を読み込む。
    無い or 空なら、フォールバックのプロンプトを返す。
    """
    path = DATA_DIR / "system_prompt.txt"
    if path.exists():
        txt = path.read_text(encoding="utf-8").strip()
        if txt:
            return txt

    # フォールバック
    return (
        "あなたは社内ヘルプデスク向けのアシスタントです。常に日本語で丁寧に回答してください。\n"
        "次のローカルナレッジがあれば、できるだけ優先して活用してください。\n"
        "ナレッジに無い内容について聞かれた場合は、その旨を伝えた上で、"
        "一般論として答えられる範囲で補足してください。"
    )


# ---------------------------------------------------------
# ローカルナレッジ読み込み (data/knowledge.txt)
#   空行で区切ってドキュメント単位に分割
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_knowledge() -> List[str]:
    path = DATA_DIR / "knowledge.txt"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def get_knowledge_docs() -> List[str]:
    return load_knowledge()


# ---------------------------------------------------------
# Streamlit セッション状態
# ---------------------------------------------------------
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    if "history" not in st.session_state:
        # [{"user": "...", "assistant": "..."}, ...]
        st.session_state.history: List[Dict[str, str]] = []

    if "loaded_log_name" not in st.session_state:
        st.session_state.loaded_log_name: str | None = None

    # session_id は get_session_id() 側で初期化


def add_history(user: str, assistant: str) -> None:
    """現在のセッション履歴 & Chat UI 両方に追加"""
    st.session_state.history.append({"user": user, "assistant": assistant})
    st.session_state.messages.append({"role": "user", "content": user})
    st.session_state.messages.append({"role": "assistant", "content": assistant})


def get_history() -> List[Dict[str, str]]:
    return st.session_state.history


# ---------------------------------------------------------
# 埋め込み & コサイン類似度検索
# ---------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """knowledge.txt の各ドキュメントを埋め込む"""
    client = get_embedding_client()
    if isinstance(client, str):
        # エラー文字列が返ってきた場合
        raise RuntimeError(client)

    if not texts:
        return []

    resp = client.embeddings.create(
        model=LOCALLM_EMBEDDING_MODEL,
        input=texts,
    )
    vectors: List[List[float]] = [d.embedding for d in resp.data]
    return vectors


def embed_query(text: str) -> List[float]:
    """クエリを埋め込む"""
    client = get_embedding_client()
    if isinstance(client, str):
        raise RuntimeError(client)

    resp = client.embeddings.create(
        model=LOCALLM_EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = 0.0
    s1 = 0.0
    s2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        s1 += a * a
        s2 += b * b
    if s1 == 0 or s2 == 0:
        return 0.0
    return dot / (math.sqrt(s1) * math.sqrt(s2))


@st.cache_resource(show_spinner=True)
def prepare_corpus_for_embeddings() -> Tuple[List[str], List[List[float]]]:
    """
    knowledge.txt を読み込み、埋め込みベクトルを構築してキャッシュ。
    """
    docs = get_knowledge_docs()
    if not docs:
        return [], []

    vectors = embed_texts(docs)
    return docs, vectors


def search_by_embedding(query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
    """
    埋め込みで類似ドキュメントを検索
    戻り値: (docs, scores)
    """
    docs, vectors = prepare_corpus_for_embeddings()
    if not docs or not vectors:
        return [], []

    q_vec = embed_query(query)
    scored: List[Tuple[float, str]] = []

    for doc, v in zip(docs, vectors):
        score = cosine_similarity(q_vec, v)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    top_docs = [d for _, d in top]
    top_scores = [s for s, _ in top]
    return top_docs, top_scores


# ---------------------------------------------------------
# LLM 呼び出し（チャット）
# ---------------------------------------------------------
def call_llm_with_context(query: str, contexts: List[str]) -> str:
    client = get_chat_client()
    if isinstance(client, str):
        # エラーメッセージが返ってきた場合
        return client

    history = get_history()

    # コンテキスト結合
    if contexts:
        context_text = "\n\n---\n\n".join(contexts)
    else:
        context_text = "ローカルナレッジ（knowledge.txt）から関連情報は見つかりませんでした。"

    # system_prompt.txt の内容 + ローカルナレッジを結合
    base_system_prompt = load_system_prompt()
    system_content = (
        f"{base_system_prompt}\n\n"
        "-----\n"
        "以下はローカルナレッジ（knowledge.txt から抽出された関連情報）です。"
        "必要に応じて参照してください。\n\n"
        f"{context_text}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    # 直近 5 ターン分の履歴を追加
    for turn in history[-5:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=LOCALLM_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = resp.choices[0].message.content or ""
    return answer


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Locallm Embedding Search",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Locallm 埋め込み検索版 💬")
    st.caption("knowledge.txt を埋め込みベクトルで検索して回答するデモ")

    init_session_state()

    # 事前にナレッジ読み込み（件数だけ出す）
    docs = get_knowledge_docs()
    doc_count = len(docs)

    # -----------------------------
    # サイドバー
    # -----------------------------
    with st.sidebar:
        # 新規チャット
        if st.button("新規チャット", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.loaded_log_name = None
            # session_id は再生成
            if "session_id" in st.session_state:
                del st.session_state["session_id"]
            st.success("新しいチャットを開始しました。")
            st.rerun()

        st.markdown("---")

        # ログ履歴
        st.subheader("履歴")
        log_files = list_log_files()
        if not log_files:
            st.caption("logs フォルダにまだログがありません。")
        else:
            st.caption("直近 20 件")
            for log_path in log_files[:20]:
                label = log_path.stem  # 例: 20251126_xxxxxxxx
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(label)
                with col2:
                    if st.button("→", key=f"load_log_{label}"):
                        history = load_history_from_log(log_path)
                        st.session_state.history = history
                        # Chat UI 用 messages を再構築
                        st.session_state.messages = []
                        for turn in history:
                            st.session_state.messages.append(
                                {"role": "user", "content": turn["user"]}
                            )
                            st.session_state.messages.append(
                                {"role": "assistant", "content": turn["assistant"]}
                            )
                        st.session_state.loaded_log_name = label
                        st.success(f"{label} の履歴を読み込みました。")
                        st.rerun()

        if st.session_state.loaded_log_name:
            st.info(f"読み込み中のログ: {st.session_state.loaded_log_name}")

        st.markdown("---")

        # ローカルナレッジ概要
        st.header("ローカルナレッジ")
        st.write(f"knowledge.txt の文書数: **{doc_count}** 件")

        knowledge_path = DATA_DIR / "knowledge.txt"
        st.caption("knowledge.txt Path")
        st.code(str(knowledge_path), language="text")
        if knowledge_path.exists() and doc_count > 0:
            st.caption("knowledge.txt 由来ドキュメントの一例（冒頭100文字）")
            st.write(docs[0][:100])

        st.markdown("---")

        system_prompt_path = DATA_DIR / "system_prompt.txt"
        st.caption("system_prompt.txt Path")
        st.code(str(system_prompt_path), language="text")
        if system_prompt_path.exists():
            try:
                sp_text = system_prompt_path.read_text(encoding="utf-8").strip()
                if sp_text:
                    st.caption("system_prompt.txt 冒頭100文字")
                    st.write(sp_text[:100])
                else:
                    st.caption("system_prompt.txt は空です。")
            except Exception as e:
                st.caption(f"system_prompt.txt の読み込みに失敗しました: {e}")
        else:
            st.caption("system_prompt.txt が存在しません。")

        st.markdown("---")

        st.subheader("環境情報")
        st.write(f"Chat Base URL: `{LOCALLM_BASE_URL}`")
        st.write(f"Chat Model: `{LOCALLM_CHAT_MODEL}`")
        st.write(f"Embedding Base URL: `{EMB_BASE_URL or 'https://api.openai.com/v1'}`")
        st.write(f"Embedding Model: `{LOCALLM_EMBEDDING_MODEL}`")

    # -----------------------------
    # これまでのメッセージ表示
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # チャット入力
    # -----------------------------
    query = st.chat_input("質問を入力してください（knowledge.txt の内容に関する質問など）")

    if query:
        # ユーザー入力表示
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # ローカルナレッジ検索（埋め込み）
        with st.spinner("埋め込みインデックスを使ってローカルナレッジを検索しています..."):
            try:
                contexts, scores = search_by_embedding(query, top_k=3)
            except RuntimeError as e:
                # クライアント設定系のエラーなど
                error_msg = str(e)
                with st.chat_message("assistant"):
                    st.error(error_msg)
                return

        # LLM 呼び出し
        with st.spinner("LLM に問い合わせ中..."):
            answer = call_llm_with_context(query, contexts)

        # アシスタント回答表示
        with st.chat_message("assistant"):
            st.write(answer)

            # 🔍 今回参照したローカルナレッジ + スコア表示
            if contexts:
                with st.expander("今回参照したローカルナレッジ（knowledge.txt, 埋め込み検索）"):
                    for i, (ctx, sc) in enumerate(zip(contexts, scores), start=1):
                        st.markdown(f"**Doc {i} (score={sc:.3f})**")
                        st.write(ctx)
            else:
                st.caption("knowledge.txt から関連する文書が見つかりませんでした。")

        # セッション履歴 & ログ保存
        add_history(query, answer)
        try:
            log_interaction(
                question=query,
                answer=answer,
                contexts=contexts,
                extra={"scores": scores},
            )
        except Exception:
            # ログ失敗でアプリが落ちないように
            pass


if __name__ == "__main__":
    main()
