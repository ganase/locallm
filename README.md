# 📘 Locallm — LLM + Local Knowledge Assistant  
**A fully local, secure, zero-server knowledge search & Q&A system**

ローカル PC 上で動作する **セキュアなナレッジ検索・Q&A アプリ**です。  
クラウド不要で業務ナレッジを安全に扱える、小規模チーム向けのローカル LLM クライアントです。

- ✔ **完全ローカル動作（LLM への API 呼び出しのみ）**
- ✔ **ナレッジはローカルファイル（knowledge.txt）**
- ✔ **Miniforge + 仮想環境を自動セットアップ**
- ✔ **会話ログはセッション単位で JSONL 保存**
- ✔ **system_prompt.txt で挙動を自由に変えられる**

---

# 🔧 1. 前提条件

| 項目 | 内容 |
|---|---|
| OS | Windows 10 / 11 |
| Python環境 | **Miniforge3 が必須**（`C:\Users\<UserName>\miniforge3` にインストール済み） |
| ネットワーク要件 | 使用する LLM API への HTTPS 接続が可能なこと |
| 配布方法 | Locallm.zip を **C:\TMP\** の下に展開 |

---

# 📦 2. セットアップ（一般ユーザー）

### **手順 1：Locallm.zip を解凍**

```
C:\TMP\Locallm\
```

---

### **手順 2：setup.bat を実行**

1. `Locallm\setup_locallm_wizard.bat` を右クリック → **管理者として実行**
2. 自動で以下を実施：

- Miniforge の存在チェック  
- Locallm 配置確認  
- `.venv` 仮想環境の作成（uv）  
- 必要パッケージのインストール  
- **デスクトップに Locallm ショートカット作成**

---

### **手順 3：デスクトップのショートカットで起動**

- ファイル名：`Locallm.lnk`
- ダブルクリックで Streamlit アプリが起動

---

# 🧭 3. ディレクトリ構成

```
Locallm/
│─ app/
│    └─ app_kwm.py
│
│─ data/
│    ├─ knowledge.txt
│    ├─ system_prompt.txt
│    └─ uploads/
│
│─ logs/
│    └─ YYYYMMDD_xxxxx.jsonl
│
│─ .env
│─ requirements.txt
│─ run_app_kwm.bat
│─ setup_venv.bat
│─ setup.bat
│─ README.md
```

---

# 💬 4. 使い方（アプリ仕様）

### ✔ 質問入力  
ナレッジ検索 → LLM 回答を返します。

### ✔ ナレッジ検索  
`knowledge.txt` を空行区切りで複数ドキュメントとして扱い、  
**キーワード一致スコアで検索**します。

### ✔ system_prompt  
`system_prompt.txt` が毎回 system role として読み込まれます。

### ✔ ログ保存  
会話ログは `logs/` フォルダに  
**1 セッション＝1 ファイル（JSONL）** で保存。

---

# 📝 5. ナレッジファイルの書き方

```
【社内規程A】
内容…

（空行）

【社内規程B】
内容…
```

---

# 🔐 6. API設定 (.env)

```
LOCALLM_BASE_URL=https://xxxxx
LOCALLM_CHAT_MODEL=xxxxx
LOCALLM_API_KEY=xxxxx
```

---

# 🧪 7. 手動起動（開発者向け）

```
cd C:\TMP\Locallm
.
un_app_kwm.bat
```

または

```
uv run streamlit run app/app_kwm.py
```

---

# 🚑 8. トラブルシューティング

### setup wizard が消える  
PowerShell で実行：

```
cd C:\TMP\Locallm
.\setup.bat
```

### Miniforge が見つからない  
以下の存在を確認：

```
C:\Users\<UserName>\miniforge3\Scripts\activate.bat
```

---

# 📄 9. ライセンス（MIT License）

```
MIT License

Copyright (c) 2025 Locallm
(略)
```

---

# 🚀 10. 今後の拡張予定

- CSV / PDF のナレッジ取り込み
- OpenAI / Azure / local embedding の併用
- 添付ファイル → 自動ナレッジ化
- UI テーマ
- マルチユーザー対応
