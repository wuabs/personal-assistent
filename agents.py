from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.tools import DuckDuckGoSearchRun, Tool

from tools import calculator_tool, summarization_tool, critique_tool

# 🔹 Модель (можно заменить на HuggingFaceHub при желании)
llm = ChatOpenAI(temperature=0.5)

# 🔍 Агент: Поисковик
search_tool = DuckDuckGoSearchRun()
search = Tool(
    name="Search",
    func=search_tool.run,
    description="Получи актуальную информацию из интернета по теме"
)

# 🧮 Агент: Калькулятор
calc = calculator_tool

# 🧾 Агент: Обобщатель текста
summarizer = summarization_tool

# 🧠 Агент: Анализ действий
critic = critique_tool

# 📚 Список всех инструментов
tools = [search, calc, summarizer, critic]
