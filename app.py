import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ローカル実行時は .env から OPENAI_API_KEY を読み込む
# (Streamlit Community Cloud では Secrets に設定した値が環境変数として渡される)
load_dotenv()


# ---------------------------------------------------------------------------
# 専門家の定義
# ラジオボタンの選択値をキーとして、LLM に渡すシステムメッセージを切り替える
# ---------------------------------------------------------------------------
EXPERTS = {
    "健康・栄養の専門家": (
        "あなたは管理栄養士の資格を持つ健康・栄養の専門家です。"
        "食事、栄養バランス、生活習慣、運動に関する質問に対して、"
        "科学的根拠に基づいた分かりやすいアドバイスを日本語で行ってください。"
        "医療的な診断が必要な内容については、医療機関の受診を勧めてください。"
    ),
    "旅行プランナー": (
        "あなたは経験豊富なプロの旅行プランナーです。"
        "旅行先の魅力、観光スポット、モデルコース、予算の目安、季節ごとのおすすめなどを、"
        "相談者の希望に合わせて具体的に提案してください。"
        "回答は日本語で、旅行が楽しみになるような前向きな表現でお願いします。"
    ),
    "キャリアアドバイザー": (
        "あなたは多くの転職・キャリア相談を手がけてきたキャリアアドバイザーです。"
        "仕事の悩み、転職、スキルアップ、面接対策などについて、"
        "相談者の状況を尊重しながら現実的で実践的なアドバイスを日本語で行ってください。"
    ),
}


# ---------------------------------------------------------------------------
# 会話履歴の管理
# st.session_state に {"role": "user" | "assistant", "expert": 専門家名, "content": 本文}
# の辞書をリストで積み上げていく
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_history_messages() -> list:
    """会話履歴を LangChain のメッセージ (HumanMessage / AIMessage) のリストに変換する。"""
    history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    return history


def get_llm_response(input_text: str, expert_type: str) -> str:
    """入力テキストと専門家の種類を受け取り、LLM の回答を文字列で返す。

    これまでの会話履歴 (st.session_state.messages) も合わせて LLM に渡すため、
    前の発言を踏まえた回答が得られる。

    Args:
        input_text: 入力フォームから送信されたテキスト
        expert_type: ラジオボタンで選択された専門家の種類 (EXPERTS のキー)

    Returns:
        LLM からの回答テキスト
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=EXPERTS[expert_type]),
        *build_history_messages(),
        HumanMessage(content=input_text),
    ]

    result = llm.invoke(messages)
    return result.content


# ---------------------------------------------------------------------------
# 画面の構築
# ---------------------------------------------------------------------------
st.set_page_config(page_title="専門家に相談できる LLM アプリ", page_icon="💬")

st.title("💬 専門家に相談できる LLM アプリ")

st.markdown(
    """
### アプリの概要
このアプリでは、LangChain を通じて LLM (OpenAI の gpt-4o-mini) に質問や相談ができます。
ラジオボタンで選んだ専門家として LLM が振る舞い、その分野の視点から回答します。
会話履歴は画面に残り、LLM はこれまでのやり取りを踏まえて回答します。

### 操作方法
1. 下のラジオボタンから、相談したい **専門家の種類** を選択します。
2. 入力フォームに **質問・相談内容** を入力します。
3. **「送信」ボタン** を押すと、LLM からの回答が画面に表示されます。
4. 続けて質問すると、前のやり取りを踏まえた回答が返ってきます。
   会話をやり直したいときは **「会話履歴をクリア」** を押してください。
"""
)

st.divider()

selected_expert = st.radio(
    "相談したい専門家を選択してください。",
    list(EXPERTS.keys()),
    horizontal=True,
)

with st.form("input_form", clear_on_submit=True):
    input_text = st.text_area(
        "質問・相談内容を入力してください。",
        placeholder="例）在宅勤務が続いて運動不足です。手軽にできる対策はありますか？",
        height=150,
    )
    submitted = st.form_submit_button("送信")

if submitted:
    if not input_text.strip():
        st.warning("質問・相談内容を入力してください。")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY が設定されていません。"
            "ローカルでは .env ファイルに、Streamlit Community Cloud では Secrets に設定してください。"
        )
    else:
        with st.spinner(f"{selected_expert}が回答を考えています..."):
            try:
                answer = get_llm_response(input_text, selected_expert)
            except Exception as e:
                st.error(f"回答の取得中にエラーが発生しました: {e}")
            else:
                # 回答が得られた場合のみ、質問と回答を会話履歴に追加する
                st.session_state.messages.append(
                    {"role": "user", "expert": selected_expert, "content": input_text}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "expert": selected_expert, "content": answer}
                )

# ---------------------------------------------------------------------------
# 会話履歴の表示
# ---------------------------------------------------------------------------
if st.session_state.messages:
    st.divider()
    st.subheader("🗂️ 会話履歴")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.caption(f"🧑‍🏫 {message['expert']}からの回答")
            st.write(message["content"])

    if st.button("会話履歴をクリア"):
        st.session_state.messages = []
        st.rerun()
