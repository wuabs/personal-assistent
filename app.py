from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from tools import get_tools

# Глобальные переменные
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_history = []
agent = None

# Функция выбора модели
def get_llm(model_choice):
    if model_choice == "OpenAI GPT":
        return ChatOpenAI(model="gpt-4o", temperature=0.4)
    elif model_choice == "HuggingFace: flan-t5-xl":
        return HuggingFaceEndpoint(repo_id="google/flan-t5-xl", temperature=0.4)
    elif model_choice == "HuggingFace: Mistral-7B":
        return HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.4)
    return ChatOpenAI(temperature=0.4)

# Инициализация агента
def init_agent(model_choice):
    global agent, memory, chat_history
    memory.clear()
    chat_history = []
    llm = get_llm(model_choice)
    tools = get_tools(llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )
    return f"Модель '{model_choice}' загружена!"

# Основная функция общения
def chat_with_assistant(user_input, file):
    global chat_history
    if agent is None:
        return "⚠️ Сначала выберите модель!", ""

    try:
        if file is not None:
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
            query = f"Сделай краткое обобщение следующего текста:\n\n{content}"
            result = agent.invoke(query)
        else:
            result = agent.invoke(user_input)

        if result is None:
            result = "⚠️ Модель не вернула результат"

    except StopIteration:
        result = "⚠️ Ошибка: генерация остановилась неожиданно (StopIteration)"
    except Exception as e:
        result = f"⚠️ Ошибка: {str(e)}"

    if user_input or file:
        chat_history.append(f"🧑 {user_input if user_input else '[загружен файл]'}")
        chat_history.append(f"🤖 {result}")
    return result, "\n\n".join(chat_history)


# Очистка истории
def clear_history():
    memory.clear()
    chat_history.clear()
    return "", ""

# Интерфейс Gradio
with gr.Blocks() as iface:
    gr.Markdown("### 🤖 Личный AI-Ассистент")

    model_choice = gr.Dropdown(
        choices=[
            "OpenAI GPT",
            "HuggingFace: flan-t5-xl",
            "HuggingFace: Mistral-7B"
        ],
        value="OpenAI GPT",
        label="Выберите модель"
    )
    load_btn = gr.Button("🔄 Загрузить модель")

    with gr.Row():
        user_input = gr.Textbox(lines=3, placeholder="Введите ваш запрос...", label="Ваш запрос")
        file_input = gr.File(label="Или загрузите .txt файл", file_types=[".txt"])

    output = gr.Textbox(label="Ответ ассистента")
    history = gr.Textbox(label="🕘 История диалога", lines=10)

    submit_btn = gr.Button("Отправить")
    clear_btn = gr.Button("🧹 Очистить историю")

    load_btn.click(init_agent, inputs=[model_choice], outputs=[output])
    submit_btn.click(chat_with_assistant, inputs=[user_input, file_input], outputs=[output, history])
    clear_btn.click(clear_history, outputs=[output, history])

if __name__ == "__main__":
    iface.launch()
