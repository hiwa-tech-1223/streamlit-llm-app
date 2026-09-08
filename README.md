# 専門家に相談できる LLM アプリ

Streamlit と LangChain を使った、専門家として振る舞う LLM に相談できる Web アプリです。

## 機能

- 入力フォームに入力したテキストを LangChain 経由で LLM (gpt-4o-mini) に送信し、回答を画面に表示します。
- ラジオボタンで「健康・栄養の専門家」「旅行プランナー」「キャリアアドバイザー」を選ぶと、
  選択に応じてシステムメッセージが切り替わり、その分野の専門家として LLM が回答します。
  専門分野以外の質問には回答せず、該当する専門家を選び直すよう案内します。
- 会話履歴を `st.session_state` で管理し、画面に表示するとともに LLM にも渡すため、
  前のやり取りを踏まえた回答が返ります。「会話履歴をクリア」ボタンでリセットできます。
  別の専門家を選んで送信した場合も、会話履歴は自動的にリセットされます。

## ローカルでの実行

```bash
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-...' > .env
streamlit run app.py
```

## Streamlit Community Cloud へのデプロイ

1. このリポジトリを GitHub に push します。
2. Streamlit Community Cloud で「New app」→ 本リポジトリ / `main` / `app.py` を指定します。
3. **Advanced settings** で Python version を **3.11** に設定します。
4. Secrets に以下を設定します。

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

5. Deploy を押します。
