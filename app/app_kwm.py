import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import uuid  # ★ 追加：セッション識別用

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# .env 読み込み & 環境変数
# ---------------------------------------------------------
load_dotenv()

LOCALLM_API_KEY = os.getenv("LOCALLM_API_KEY")
LOCALLM_BASE_URL = os.getenv(
    "LOCALLM_BASE_URL",
    "",
)
LOCALLM_CHAT_MODEL = os.getenv("LOCALLM_CHAT_MODEL", "")

# ---------------------------------------------------------
# パス設定
# app_kwm.py は app/ 配下にある想定
# ルート:
#   LOCALLMAI_LILT/
#     app/app_kwm.py
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
#   ⚠ 日付 + ランダム文字列で「セッションごと」のログファイルに追記
# ---------------------------------------------------------
def log_interaction(
    question: str,
    answer: str,
    contexts: List[str],
    extra: Dict[str, Any] | None = None,
) -> None:
    """logs/ 内の <日付>_<ランダム>.jsonl に Q&A とコンテキストを追記"""
    extra = extra or {}

    # セッションごとに 1 つのログファイル名を持つ
    log_name = st.session_state.get("log_file_name")
    if not log_name:
        date_str = datetime.now().strftime("%Y%m%d")
        rand = uuid.uuid4().hex[:8]
        log_name = f"{date_str}_{rand}.jsonl"
        st.session_state.log_file_name = log_name

    log_path = LOGS_DIR / log_name

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
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
    """logs/YYYYMMDD_xxxxxxxx.jsonl から history を組み立てる"""
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
# LOCALLM AI クライアント
# ---------------------------------------------------------
def get_client():
    """LOCALLM AI 用 OpenAI 互換クライアントを返す（エラー時は str を返す）"""
    if not LOCALLM_API_KEY:
        return "LOCALLM_API_KEY が設定されていません。.env を確認してください。"

    client = OpenAI(
        api_key=LOCALLM_API_KEY,
        base_url=LOCALLM_BASE_URL,
    )
    return client


# ---------------------------------------------------------
# system_prompt.txt 読み込み
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    """
    data/system_prompt.txt の内容を読み込む。
    無い or 空なら、従来の固定プロンプトをデフォルトとして返す。
    """
    path = DATA_DIR / "system_prompt.txt"
    if path.exists():
        txt = path.read_text(encoding="utf-8").strip()
        if txt:
            return txt

    # フォールバック用（いままでハードコードしていた内容）
    return (
        "あなたはだからこそ生命保険向けの社内ヘルプデスクAIです。常に日本語で丁寧に回答してください。\n"
        "次のローカルナレッジがあれば、できるだけ優先して活用してください。\n"
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
                header = next(reader, None)  # 1行目ヘッダー想定
                for row in reader:
                    line = ", ".join(col.strip() for col in row if col.strip())
                    if line:
                        docs.append(line)
        except Exception:
            continue

    return docs


def get_knowledge_docs() -> List[str]:
    """knowledge.txt + uploads 内ファイルをまとめたドキュメント一覧を返す"""
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

    # ★ 追加：このセッションで使うログファイル名
    if "log_file_name" not in st.session_state:
        st.session_state.log_file_name: str | None = None


def add_history(user: str, assistant: str) -> None:
    """現在のセッション履歴 & Chat UI 両方に追加"""
    st.session_state.history.append({"user": user, "assistant": assistant})
    st.session_state.messages.append({"role": "user", "content": user})
    st.session_state.messages.append({"role": "assistant", "content": assistant})


def get_history() -> List[Dict[str, str]]:
    return st.session_state.history


# ---------------------------------------------------------
# シンプルなキーワード検索
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """超簡易トークナイズ（空白と一部記号 + 日本語のお決まりフレーズで分割）"""

    # ① 日本語クエリでよく付けるフレーズをあらかじめスペースに置き換える
    jp_phrases = [
        "について教えて",
        "について",
        "とは何ですか",
        "とはなんですか",
        "とは？",
        "とは?",
        "とは",
        "って何",
        "ってなに",
        "のことを教えて",
        "のこと教えて",
        "のこと",
        "を教えて",
    ]
    for p in jp_phrases:
        text = text.replace(p, " ")

    # ② 記号類でスペース区切りにする（既存ロジック）
    seps = " \t\r\n、。・，．「」『』()（）[]【】：:；;!?！？"
    for ch in seps:
        text = text.replace(ch, " ")

    tokens = [t for t in text.lower().split(" ") if t]

    # ③ 記号だけ・1文字すぎるものをざっくり除去（英数字向けの簡易フィルタ）
    cleaned = []
    for t in tokens:
        # ひらがな・カタカナ・漢字が含まれている場合は長さ1でも残す
        if any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in t):
            cleaned.append(t)
        else:
            if len(t) >= 2:
                cleaned.append(t)

    return cleaned


def search_knowledge(query: str, docs: List[str], top_k: int = 3) -> List[str]:
    """Jaccard + 部分一致ボーナスでシンプルにスコアリング"""

    if not docs:
        return []

    # 質問側のトークン
    q_tokens = tokenize(query)
    # 長さ 1 の記号っぽいものは雑に捨てる
    q_tokens = [t for t in q_tokens if len(t) >= 2]

    # それでも何も残らなければ、生のクエリをそのまま 1 個だけ使う
    if not q_tokens:
        q_tokens = [query.strip()] if query.strip() else []

    if not q_tokens:
        return []

    scored: List[tuple[float, str]] = []

    for doc in docs:
        doc_text = doc  # 日本語なので lower() はあまり意味なし

        # 1) Jaccard ベースのスコア
        base_score = 0.0
        d_tokens = tokenize(doc)
        if d_tokens:
            q_set = set(q_tokens)
            d_set = set(d_tokens)
            inter = len(q_set & d_set)
            union = len(q_set | d_set)
            if union > 0:
                base_score = inter / union  # 0.0〜1.0

        # 2) 部分一致ボーナス（どれか 1 キーワードでも含まれていれば 0.5 加点）
        substr_bonus = 0.0
        for kw in q_tokens:
            if kw and kw in doc_text:
                substr_bonus = 0.5
                break

        score = base_score + substr_bonus

        if score > 0:
            scored.append((score, doc))

    # スコア順に並び替え
    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------
# LLM 呼び出し
# ---------------------------------------------------------
def call_local_llm(query: str, contexts: List[str]) -> str:
    client = get_client()
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
        "以下はローカルナレッジ（knowledge.txt / uploads から抽出された関連情報）です。"
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
    st.set_page_config(page_title="Keyword match LLM", page_icon="💬", layout="wide")
    st.title("Locallm💬")
    st.caption(
        "Locallm - Keyword match LLM - Built with Streamlit, a product of Knock Knock Inc. "
        "For internal, local-only purposes. Not utilized for LLM training datasets or models."
    )
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
            st.session_state.log_file_name = None  # ★ 追加：新しいセッションなのでログファイル名もリセット
            st.success("新しいチャットを開始しました。")
            st.rerun()

        st.markdown("---")

        # ログ履歴
        st.subheader("履歴")
        log_files = list_log_files()
        if not log_files:
            st.caption("logs フォルダにまだログがありません。")
        else:
            st.caption("直近20 件")
            for log_path in log_files[:20]:
                label = log_path.stem  # 例: 20251126_ab12cd34
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
                        # ★ 追加：このログに追記したい場合に備えて、ファイル名も保持
                        st.session_state.log_file_name = log_path.name
                        st.success(f"{label} の履歴を読み込みました。")
                        st.rerun()

        if st.session_state.loaded_log_name:
            st.info(f"読み込み中のログ: {st.session_state.loaded_log_name}")

        st.markdown("---")

        # 添付ファイル UI は一旦封印中（将来使うならコメントアウト解除）
        # st.subheader("添付ファイル追加 β")
        # ...

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
        st.write(f"Base URL: `{LOCALLM_BASE_URL}`")
        st.write(f"Model: `{LOCALLM_CHAT_MODEL}`")

    # -----------------------------
    # これまでのメッセージ表示
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # チャット入力
    # -----------------------------
    query = st.chat_input("質問を入力してください（楽天生命の業務・社内 FAQ など）")

    if query:
        # ユーザー入力表示
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # ローカルナレッジ検索
        with st.spinner("ローカルナレッジを検索しています..."):
            contexts = search_knowledge(query, docs, top_k=3)

        # LLM 呼び出し
        with st.spinner("問い合わせ中..."):
            answer = call_local_llm(query, contexts)

        # アシスタント回答表示
        with st.chat_message("assistant"):
            st.write(answer)

            # 🔍 ここで「今回参照したローカルナレッジ」の表示
            if contexts:
                with st.expander("今回参照したローカルナレッジ（knowledge.txt / uploads）"):
                    for i, ctx in enumerate(contexts, start=1):
                        st.markdown(f"**Doc {i}**")
                        st.write(ctx)
            else:
                st.caption("knowledge.txt / uploads から関連する文書が見つかりませんでした。")

        # セッション履歴 & ログ保存
        add_history(query, answer)
        try:
            log_interaction(query, answer, contexts)
        except Exception:
            # ログ失敗でアプリが落ちないように
            pass


if __name__ == "__main__":
    main()
