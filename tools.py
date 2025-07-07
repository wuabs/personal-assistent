from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun

def get_tools(llm):
    # 🔧 Python REPL инструмент
    repl_tool = PythonREPL()

    # 🔧 Инструмент обобщения
    def summarize_tool(text):
        prompt = f"Сделай краткое и понятное обобщение текста:\n{text}"
        return llm.invoke(prompt)

    # 🔧 Инструмент критики
    def critic_tool(text):
        prompt = f"Проанализируй это действие или поведение: {text}"
        return llm.invoke(prompt)

    # 🔧 Поисковик
    duckduckgo_tool = DuckDuckGoSearchRun()
    

    return [
        Tool(
            name="Summarizer",
            func=summarize_tool,
            description="Обобщает текст пользователя"
        ),
        Tool(
            name="Critic",
            func=critic_tool,
            description="Анализирует и оценивает действия, даёт рекомендации"
        ),
        Tool(
            name="PythonREPL",
            func=repl_tool.run,
            description="Выполняет математические и логические вычисления",
            return_direct=False
        ),
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_tool.run,
            description="Поиск информации в интернете"
        )
    ]
