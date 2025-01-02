content_summary_prompt_template = """
    Dựa trên các tài liệu sau đây, hãy tổng hợp thông tin quan trọng liên quan đến câu hỏi của user: {question}
    Chỉ cần lấy các nội dung liên quan đến câu hỏi của user từ các tài liệu, không thêm thông tin bên ngoài.
    Nếu không có thông tin liên quan đến câu hỏi của user trong các tài liệu thì trả về không có thông tin.

    Các tài liệu:
    {documents}

    Các thông tin liên quan đến câu hỏi của user:
"""

history_summary_prompt_template = """
    Dựa trên lịch sử cuộc trò chuyện sau, hãy tóm tắt những ý chính của cả user lẫn assistant.
    Chỉ cần tóm tắt các nội dung quan trọng, không thêm thông tin ngoài những gì được cung cấp.

    Lịch sử cuộc trò chuyện:
    {history}

    Tóm tắt:
"""

history_retriever_prompt_template = """
    Dựa trên lịch sử cuộc trò chuyện sau, hãy lấy ra các thông tin liên quan đến câu hỏi của user: {question}.
    Chỉ cần lấy các nội dung liên quan, nếu không có thông tin liên quan trong lịch sử trò truyện hãy trả lời lịch sử không có thông tin liên quan đến câu hỏi.

    Lịch sử cuộc trò chuyện:
    {history}

    Các thông tin liên quan đến câu hỏi của user:
"""

tool_usage_prompt_template = """
    Bạn là một Agent thông minh. Nếu câu hỏi của user yêu cầu thông tin có thể lấy từ các tool dưới đây, hãy sử dụng tool tương ứng.

    Danh sách tools:
    {tools}

    Khi trả lời, chỉ cần trả về : {{"tool_name": "<tên_tool>", "arguments": {{<các arguments>}}}}
    Nếu không sử dụng tool, trả về: {{}}

    Ví dụ:
    - "Thời tiết Hồ Chí Minh thế nào?" -> {{"tool_name": "get_weather", "arguments": {{"location": "Hồ Chí Minh"}}}}
    - "Tính diện tích với cao 20 và dài 30" -> {{"tool_name": "calculate_area", "arguments": {{"width": 30, "height": 20}}}}
    - "Bạn khỏe không?" -> {{}}

    Câu hỏi của user: {question}
    Trả lời:
"""

qa_system_prompt_template = """
    Bạn là một trợ lý thông minh tên là Toka. Hãy trả lời câu hỏi sau, khi trả lời hãy làm theo thứ tự ưu tiên sau:

    1. Nếu có thông tin từ công cụ hãy sử dụng.
    2. Nếu không, hãy sử dụng thông tin từ dữ liệu.
    3. Sử dụng thông tin từ lịch sử trò chuyện nếu cần.
    3. Nếu không có thông tin nào, hãy trả lời một cách chính xác và rõ ràng.

    Câu hỏi: {question}

    Thông tin từ công cụ: {tool_response}
    Thông tin từ dữ liệu: {context}
    Thông tin lịch sử trò chuyện: {chat_history}

    Câu trả lời:
"""