from dotenv import load_dotenv
load_dotenv()

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from tools import get_tools 
import warnings
warnings.filterwarnings("ignore")

# Функция выбора модели
def get_llm(model_choice):
    if model_choice == "OpenAI GPT":
        return ChatOpenAI(model="gpt-4o", temperature=0.4)
    elif model_choice == "HuggingFace: flan-t5-xl":
        return HuggingFaceEndpoint(repo_id="google/flan-t5-xl", temperature=0.4)
    elif model_choice == "HuggingFace: Mistral-7B":
        return HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.4)
    return ChatOpenAI(temperature=0.4)

agent = None
chat_history = []
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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


def run_assistant():
    print("🤖 Личный ассистент активирован!")

    print("\nДоступные модели:")
    print("1. OpenAI GPT")
    print("2. HuggingFace: flan-t5-xl")
    print("3. HuggingFace: Mistral-7B")
    choice_map = {
        "1": "OpenAI GPT",
        "2": "HuggingFace: flan-t5-xl",
        "3": "HuggingFace: Mistral-7B"
    }
    choice = input("\nВыберите модель (1/2/3): ").strip()
    model_choice = choice_map.get(choice, "OpenAI GPT")
    init_agent(model_choice)

    print("\n🔹 Введите ваш запрос. Для выхода введите 'exit'")
    print("Примеры:\n- Найди информацию о ИИ в образовании\n- Обобщи текст: ...\n- Что ты думаешь о том, что я весь день прокрастинировала?\n- Сколько будет 15% от 8200?\n")

    while True:
        try:
            user_input = input("🗣 Ввод: ")
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("👋 До встречи!")
                break
            response = agent.invoke(user_input)
            print(f"\n🤖 Ответ:\n{response}\n")
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")

if __name__ == "__main__":
    run_assistant()
