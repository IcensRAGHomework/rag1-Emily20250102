import json
import traceback
import requests
import base64

from mimetypes import guess_type

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def getLLM():
    gpt_chat_version = 'gpt-4o'
    gpt_config = get_model_configuration(gpt_chat_version)

    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    return llm

def get_returnHolidyStruct():
    response_schemas = [
        ResponseSchema(
            name="Result",
            description="請提供包含紀念日的時間與紀念日名稱的 JSON 物件"),
        ResponseSchema(
            name="date",
            description="紀念日的時間",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="紀念日名稱",
            type="str")]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # print(format_instructions)
    return format_instructions

def generate_hw01(question):
    llm = getLLM()
    
    get_response_format = get_returnHolidyStruct()

    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題的所有答案,{response_format}"),
        # ("system","使用台灣語言並回答問題"),
        ("human","{question}")
        ])
    prompt = prompt.partial(response_format=get_response_format)
    response = llm.invoke(prompt.format(question=question)).content
    # return response

    examples = [
        {"input": """```json
                    {
                            "Result": [
                                    content
                            ]
                    }
                    ```""",
        "output": """{
                            "Result": [
                                    content
                            ]
                    }"""}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", "將我提供的文字進行處理"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response2 = llm.invoke(prompt2.invoke(input = response)).content
    return response2

    
# answer = generate_hw01("2024年台灣10月紀念日有哪些?")
# print(answer)


def get_returnHolidyStruct(data: str):
    response_schemas = [
        ResponseSchema(
            name="Result",
            description="請提供包含紀念日的時間與紀念日名稱的 JSON 物件"),
        ResponseSchema(
            name="date",
            description="紀念日的時間",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="紀念日名稱",
            type="str")]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system","將我提供的資料整理成指定格式,使用台灣語言,{format_instructions}"),
        ("human","{data}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = getLLM().invoke(prompt.format_messages(data=data)).content
    return response

def get_holiday_info(year: str, month: str) -> str:
    api_key = "0mdLoL5bl0dsgoPA2cpEITHk7vf1fxf9"
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    responseFromCalendarific = requests.get(url)
    responseFromCalendarific = responseFromCalendarific.json()
    responseFromCalendarific = responseFromCalendarific.get('response')
    return responseFromCalendarific

class GetHolidayDate(BaseModel):
    year: str = Field(description="holiday name")
    month: str = Field(description="holiday date in ISO")

getHolidayTool = StructuredTool.from_function(
    name="getHoliday",
    description="get holiday info from calendarific response",
    func=get_holiday_info,
    args_schema=GetHolidayDate,
) 


def generate_hw02(question):
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [getHolidayTool]
    agent = create_openai_functions_agent(getLLM(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input":question}).get('output')

    get_response_format = get_returnHolidyStruct(response)

    examples = [
        {"input": """```json
                    {
                            "Result": [
                                    content
                            ]
                    }
                    ```""",
        "output": """{
                            "Result": [
                                    content
                            ]
                    }"""}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
    )
    prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "將我提供的文字進行處理，結果不要出現```與json"),
        few_shot_prompt,
        ("human", "{input}"),
        ]
    )
    response2 = getLLM().invoke(prompt2.invoke(input = get_response_format)).content
    return response2

# answer2 = generate_hw02("2024年台灣10月紀念日有哪些?")
# print(answer2)

    
def isHolidyExist(data: str):
    response_schemas = [
        ResponseSchema(
            name="Result",
            description="Json物件名稱，含有add與reason"),
        ResponseSchema(
            name="add",
            description="布林值，表示是否需要將節日新增到節日清單中。節日存在的定義是節日名稱與日期都符合才算存在，如果不存在，則為 true；否則為 false。",
            type="boolean"),
        ResponseSchema(
            name="reason",
            description="描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。",
            type="str")]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system","將我提供的資料整理成指定格式,使用台灣語言,{format_instructions}"),
        ("human","{data}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = getLLM().invoke(prompt.format_messages(data=data)).content

    # print(format_instructions)
    return response




def generate_hw03(question2, question3):

    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [getHolidayTool]

    agent = create_openai_functions_agent(getLLM(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    history = ChatMessageHistory()
    def get_history() -> ChatMessageHistory:
        return history

    agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    tools = [getHolidayTool]
    agent = create_openai_functions_agent(getLLM(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_with_chat_history.invoke({"input":question2}).get('output')

    # print(response)
    response = agent_with_chat_history.invoke({"input":question3}).get('output')

    get_response_format = isHolidyExist(response)
    print(get_response_format)

    examples = [
            {"input": """```json
                        {
                            content
                        }
                        ```""",
            "output": """{
                                "Result": {
                                        content
                                }
                        }"""}
        ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
    )
    prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "將我提供的文字進行處理，結果不要出現```與json"),
        few_shot_prompt,
        ("human", "{input}"),
        ]
    )
    response2 = getLLM().invoke(prompt2.invoke(input = get_response_format)).content
    # print(response2)
    return response2



def get_returnScoreStruct():
    response_schemas = [
        ResponseSchema(
            name="score",
            description="指定隊伍的分數",
            type="interger")]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()

    return format_instructions

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_hw04(question):
    llm = getLLM()
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    get_response_format = get_returnScoreStruct()

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "辨識圖片中的文字表格,{response_format}"),
                (
                    "user",
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    ],
                ),
                ("human", "{question}")
            ]
        )
    
    prompt = prompt.partial(response_format=get_response_format)
    response = llm.invoke(prompt.format_messages(question = question)).content

    examples = [
        {"input": """```json
                    {
                        "item" : "content"
                    }
                    ```""",
        "output": """{
                            "Result":
                                {
                                    "item" : content
                                }
                    }"""}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", "將我提供的文字進行處理，結果不要出現```與json"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response3 = llm.invoke(prompt2.invoke(input = response)).content
    return response3

# print("print hw4"+generate_hw04(question))
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
# print(demo("你好，使用繁體中文 ").content)