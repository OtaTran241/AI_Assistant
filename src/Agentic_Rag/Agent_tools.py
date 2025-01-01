from typing import Callable, Any, Dict
from Features.ChatAI import get_google_search
from Features.WindowControl import open_app as oa
from Features.WindowControl import close_app as ca


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
    results = get_google_search(query)
    results = f"Kết quả của tìm kiếm trên google của: {query} là " + str(results)
    return results

def open_app(appname: str) -> str:
    state = oa(appname)
    if state:
        results = f"mở {appname} thành công."
    else:
        results = f"mở {appname} thất bại."
    return results

def close_app(appname: str) -> str:
    state = ca(appname)
    if state:
        results = f"đóng {appname} thành công."
    else:
        results = f"đóng {appname} thất bại."
    return results

open_tool = Tool(
    name="open_app",
    arguments={"appname": str},
    description="Mở ứng dụng nhất định",
    function=open_app
)

close_tool = Tool(
    name="close_app",
    arguments={"appname": str},
    description="Đóng ứng dụng nhất định",
    function=close_app
)

google_tool = Tool(
    name="get_google",
    arguments={"query": str},
    description="Tìm kiếm/Search kết quả từ google theo yêu cầu",
    function=get_google
)

tools = [google_tool, open_tool, close_tool]