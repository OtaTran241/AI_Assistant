from typing import Callable, Any, Dict
import ChatAI as ca

class Tool:
    def __init__(self, name: str, arguments: Dict[str, Any], description: str, function: Callable):
        self.name = name
        self.arguments = arguments
        self.description = description
        self.function = function

    def call(self, **kwargs):
        return self.function(**kwargs)

    def get_header(self) -> str:
        args = ", ".join([f"{k}: {v.__name__}" for k, v in self.arguments.items()])
        return f"{self.name}({args}): {self.description}"


def get_google(query: str) -> str:
    search_results = ca.get_google_search(query)
    return search_results

google_tool = Tool(
    name="get_google",
    arguments={"query": str},
    description="Tìm kiếm/Search kết quả từ google theo yêu cầu",
    function=get_google
)

tools = [google_tool]