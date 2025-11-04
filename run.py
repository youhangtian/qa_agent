import yaml

from smolagents import OpenAIServerModel, ToolCallingAgent, FinalAnswerTool
from smolagents.agents import ToolOutput, ActionOutput
from tools import TimeTool, TextToSqlTool, QueryTool, DocSearchTool

model = OpenAIServerModel(
    model_id='qwen30b',
    api_base='http://localhost:9997/v1',
    api_key='xxx',
)

# messages = [
#     {"role": "user", "content": "你是谁"}
# ]
# output = model.generate_stream(messages)
# for chunk in output:
#     print(chunk.content, end='', flush=True)

with open('cfg.yaml') as f:
    cfg = yaml.safe_load(f)

agent = ToolCallingAgent(
    tools = [
        TimeTool(),
        TextToSqlTool(),
        QueryTool(),
        DocSearchTool(),
        FinalAnswerTool()
    ],
    model=model,
    prompt_templates=cfg['prompts'],
    max_steps=10,
    verbosity_level=2,
)

agent.run('现在是几号？')
agent.run('绍兴上个月的经济情况如何？')
agent.run('越城区这个月上报了多少事件？')

# result = agent.run('绍兴上个月的经济情况', stream=True)
# for chunk in result:
#     if isinstance(chunk, ToolOutput):
#         print(f"[Tool Output] ---------------")
#         print('observation:', chunk.observation)
#         print('output:', chunk.output)
#         print('---------------')
#     elif isinstance(chunk, ActionOutput):
#         print(f"[Action Output] ---------------")
#         print('output:', chunk.output)
#         print('is_final_answer:', chunk.is_final_answer)
#         print('---------------')
#     else:
#         print('[Other Output] ---------------')
#         print('chunk type:', type(chunk))
#         print('---------------')
