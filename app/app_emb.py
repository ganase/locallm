import os
import json
import math
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

# チャット用 LLM
LLM_API_KEY = os.getenv("LOCALLM_API_KEY") or os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LOCALLM_BASE_URL") or os.getenv(
    "LLM_BASE_URL",
    "",
)
LLM_MODEL = os.getenv("LOCALLM_CHAT_MODEL") or os.getenv("LLM_MODEL", "")

# 埋め込み用（プロバイダに依存しない抽象名）
EMB_API_KEY = os.getenv("EMB_API_KEY")
EMB_BASE_URL = os.getenv("EMB_BASE_URL", "https://api.openai.com/v1")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------
# パス設定
#   Locallm/
#     app/app_emb.py
#     data/knowledge.txt
#     data/system_prompt.txt
#     data/uploads/
#     logs/
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# ログ書き込み（1行1JSON の jsonl 形式）
# ---------------------------------------------------------
def log_interaction(
    question: str,
    answer: str,
    contexts: List[str],
    extra: Dict[str, Any] | None = None,
) -> None:
    """logs/YYYYMMDD.jsonl に Q&A とコンテキストを追記"""
    extra = extra or {}
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = LOGS_DIR / f"{date_str}.jsonl"

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    record.update(extra)

    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # ログ書き込み失敗は無視（アプリが落ちないように）
        pass


def list_log_files() -> List[Path]:
    """logs/ 配下の *.jsonl を新しい順に返す"""
    files = sorted(LOGS_DIR.glob("*.jsonl"), reverse=True)
    return files


def load_history_from_log(log_path: Path) -> List[Dict[str, str]]:
    """logs/YYYYMMDD_xxxxxx.jsonl から history を組み立てる"""
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
# LLM / Embedding クライアント
# ---------------------------------------------------------
def get_llm_client():
    """チャット用 LLM クライアント"""
    if not LLM_API_KEY:
        return "LLM_API_KEY が設定されていません。.env を確認してください。"

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    return client


def get_emb_client():
    """Embedding 用クライアント（OpenAI / Azure / その他 何でも可）"""
    if not EMB_API_KEY:
        return "EMB_API_KEY が設定されていません。.env を確認してください。"

    client = OpenAI(
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )
    return client


# ---------------------------------------------------------
# system_prompt.txt 読み込み
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    """
    data/system_prompt.txt の内容を読み込む。
    無い or 空なら、デフォルトのプロンプトを返す。
    """
    path = DATA_DIR / "system_prompt.txt"
    if path.exists():
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            return txt

    # フォールバック用
    return (
        "あなたはローカルナレッジを活用する社内ヘルプデスクAIです。"
        "常に日本語で丁寧に回答してください。\n"
        "ローカルナレッジがあればできるだけ優先して活用し、"
        "ナレッジに無い内容について聞かれた場合は、その旨を伝えた上で、"
        "一般論として答えられる範囲で補足してください。"
    )


# ---------------------------------------------------------
# ローカルナレッジ読み込み
#   - data/knowledge.txt （空行区切りで 1 ドキュメント）
#   - data/uploads/*.txt, *.md （空行区切りで 1 ドキュメント）
#   - data/uploads/*.csv （1 行 = 1 ドキュメント扱い）
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_knowledge() -> List[str]:
    docs: List[str] = []

    # 1) data/knowledge.txt
    knowledge_path = DATA_DIR / "knowledge.txt"
    if knowledge_path.exists():
        text = knowledge_path.read_text(encoding="utf-8", errors="ignore")
        docs.extend(b.strip() for b in text.split("\n\n") if b.strip())

    # 2) data/uploads/*.txt, *.md
    for path in UPLOAD_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        docs.extend(b.strip() for b in text.split("\n\n") if b.strip())

    # 3) data/uploads/*.csv ･･･ 1 行 = 1 ドキュメント（ヘッダー行はスキップ）
    import csv

    for path in UPLOAD_DIR.glob("*.csv"):
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                _header = next(reader, None)
                for row in reader:
                    line = ", ".join(col.strip() for col in row if col.strip())
                    if line:
                        docs.append(line)
        except Exception:
            continue

    return docs


def get_knowledge_docs() -> List[str]:
    return load_knowledge()


# ---------------------------------------------------------
# セッション状態
# ---------------------------------------------------------
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, str]] = []

    if "loaded_log_name" not in st.session_state:
        st.session_state.loaded_log_name: str | None = None


def add_history(user: str, assistant: str) -> None:
    """現在のセッション履歴 & Chat UI 両方に追加"""
    st.session_state.history.append({"user": user, "assistant": assistant})
    st.session_state.messages.append({"role": "user", "content": user})
    st.session_state.messages.append({"role": "assistant", "content": assistant})


def get_history() -> List[Dict[str, str]]:
    return st.session_state.history


# ---------------------------------------------------------
# Embedding 関連：コサイン類似度
# ---------------------------------------------------------
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """単純なコサイン類似度計算（numpy を使わない版）"""
    if not vec_a or not vec_b:
        return 0.0

    # 長さが違う場合は短い方に合わせる
    n = min(len(vec_a), len(vec_b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        a = vec_a[i]
        b = vec_b[i]
        dot += a * b
        na += a * a
        nb += b * b

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot / (math.sqrt(na) * math.sqrt(nb))


def embed_texts(texts: List[str]) -> List[List[float]]:
    """与えられた texts を Embedding ベクトルに変換"""
    client = get_emb_client()
    if isinstance(client, str):
        # エラーメッセージの場合は例外にして上位でハンドリング
        raise RuntimeError(client)

    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=texts,
    )
    vectors: List[List[float]] = [d.embedding for d in resp.data]
    return vectors


# ---------------------------------------------------------
# コーパスのベクトル化 & インデックス構築（キャッシュ）
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_corpus_index() -> Tuple[List[str], List[List[float]]]:
    """
    knowledge.txt + uploads をまとめて読み込み、
    埋め込みベクトルに変換して保持する。
    """
    docs = get_knowledge_docs()
    if not docs:
        return [], []

    vectors = embed_texts(docs)
    return docs, vectors


def retrieve_with_embedding(query: str, top_k: int = 3) -> List[str]:
    """
    クエリを埋め込み、コサイン類似度の高い順に top_k 件返す
    """
    docs, vectors = build_corpus_index()
    if not docs or not vectors:
        return []

    # クエリを埋め込み
    q_vec = embed_texts([query])[0]

    scored: List[Tuple[float, str]] = []
    for doc, vec in zip(docs, vectors):
        score = cosine_similarity(q_vec, vec)
        if score > 0.0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]


# ---------------------------------------------------------
# LLM 呼び出し
# ---------------------------------------------------------
def call_llm_with_context(query: str, contexts: List[str]) -> str:
    client = get_llm_client()
    if isinstance(client, str):
        # エラーメッセージが返ってきた場合
        return client

    history = get_history()

    # コンテキスト結合
    if contexts:
        context_text = "\n\n---\n\n".join(contexts)
    else:
        context_text = "ローカルナレッジから関連情報は見つかりませんでした。"

    base_system_prompt = load_system_prompt()
    system_content = (
        f"{base_system_prompt}\n\n"
        "-----\n"
        "以下はローカルナレッジ（knowledge.txt / uploads）から抽出された関連情報です。"
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
        model=LLM_MODEL,
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
        page_title="Locallm Embedding版",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Locallm Embedding版 (ベクトル検索) 🧠")
    st.caption("ローカルナレッジ + Embedding による RAG テスト用アプリ")

    init_session_state()
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
                label = log_path.stem
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(label)
                with col2:
                    if st.button("→", key=f"load_log_{label}"):
                        history = load_history_from_log(log_path)
                        st.session_state.history = history
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
        st.write(f"knowledge.txt + uploads の文書数: **{doc_count}** 件")

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
        st.write(f"[LLM] Base URL: `{LLM_BASE_URL}`")
        st.write(f"[LLM] Model    : `{LLM_MODEL}`")
        st.write(f"[EMB] Base URL: `{EMB_BASE_URL}`")
        st.write(f"[EMB] Model    : `{EMB_MODEL}`")

    # -----------------------------
    # これまでのメッセージ表示
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # チャット入力
    # -----------------------------
    query = st.chat_input("質問を入力してください（ローカルナレッジ + Embedding で検索）")

    if query:
        # ユーザー入力表示
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # RAG: Embedding 検索
        with st.spinner("ローカルナレッジ（Embedding）を検索しています..."):
            try:
                contexts = retrieve_with_embedding(query, top_k=3)
            except Exception as e:
                contexts = []
                st.error(f"Embedding 検索中にエラーが発生しました: {e}")

        # LLM 呼び出し
        with st.spinner("LLM に問い合わせ中..."):
            answer = call_llm_with_context(query, contexts)

        # アシスタント回答表示
        with st.chat_message("assistant"):
            st.write(answer)

            if contexts:
                with st.expander("今回参照したローカルナレッジ（Embedding 検索結果）"):
                    for i, ctx in enumerate(contexts, start=1):
                        st.markdown(f"**Doc {i}**")
                        st.write(ctx)
            else:
                st.caption("Embedding によるローカルナレッジ検索結果はありませんでした。")

        # セッション履歴 & ログ保存
        add_history(query, answer)
        log_interaction(query, answer, contexts)


if __name__ == "__main__":
    main()
