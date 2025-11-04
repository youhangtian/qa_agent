import time
import requests
import pymysql
from smolagents import Tool 

from chroma import ChromaDBClient

model_url = 'http://localhost:9997/v1/chat/completions'

sql_prompt = '''
根据提供的数据库表结构，使用sql代码来回答用户问题。回答要求：
1,使用一段sql代码来回答用户问题，sql代码以```sql开头，以```结尾。
2,用户问题中涉及的时间，在使用sql查询时用字段"dt"来查询，不要使用其他的时间字段来查询。
3,用户问题中涉及的时间，使用CURDATE(), DATE_FORMAT(), YEAR(), MONTH(), QUARTER()等函数来查询。
4,sql代码要符合mysql语法，执行该sql代码后，得到的数据能准确回答用户问题。
```数据库表结构
{table_info}
```
```用户问题
{question}
```
'''
    
def get_cursor():
    db = pymysql.connect(
        host='localhost',
        port=3306,
        user='tyh',
        password='123456',
        database='test',
        charset='utf8'
    )
    cursor = db.cursor(cursor=pymysql.cursors.DictCursor)
    return cursor
    
class TextToSqlTool(Tool):
    name = 'text_to_sql_tool'
    description = '''将用户的自然语言问题转换为数据库的SQL查询语句，数据库里有绍兴市每天各种事件的上报情况，包括事件类型、事件数量、上报时间等信息'''
    inputs = {
        'question': {
            'type': 'string',
            'description': '用户提出的问题'
        }
    }
    output_type = 'string'

    def __init__(self):
        super().__init__()
        self.model_name = 'qwen7b'
        self.table_name = 'zkyc_event_info_dt'

    def forward(self, question: str):
        cursor = get_cursor()

        table_info = ''
        try:
            cursor.execute(f"SHOW CREATE TABLE {self.table_name}")
            result = cursor.fetchone()
            table_info = result['Create Table']
        except Exception as e:
            print(f"Error fetching table info: {e}")
            
        prompt = sql_prompt.format(table_info=table_info, question=question)
        messages = [
            {'role': 'user', 'content': prompt},
        ]
        request_data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0,
        }
        response = requests.post(model_url, json=request_data)
        output = response.json()['choices'][0]['message']['content']

        sql_string = output.split('```sql')[1].split('```')[0].strip()
        return sql_string

class QueryTool(Tool):
    name = 'query_tool'
    description = '''通过sql代码查询数据库'''
    inputs = {
        'sql_string': {
            'type': 'string',
            'description': 'sql代码'
        }
    }
    output_type = 'string'

    def __init__(self):
        super().__init__()

    def forward(self, sql_string: str):
        cursor = get_cursor()

        data = []
        try:
            cursor.execute(sql_string)
            data = cursor.fetchall()
        except Exception as e:
            print(f"Error executing SQL: {e}")

        return '\n'.join([str(item) for item in data])
    

class TimeTool(Tool):
    name = 'time_tool'
    description = '''获取当前时间'''
    inputs = {}
    output_type = 'string'

    def __init__(self):
        super().__init__()

    def forward(self):
        time_now = time.strftime('今天是%Y年%m月%d日')
        return time_now
    
class DocSearchTool(Tool):
    name = 'doc_search_tool'
    description = '''从知识库里查找出回答用户问题所需的文档信息，知识库里是绍兴市每天的城市运行情况，如经济活力、交通运行、环境质量等方面的信息'''
    inputs = {
        'question': {
            'type': 'string',
            'description': '用户问题'
        },
        'k': {
            'type': 'integer',
            'description': '返回的文档数量，默认为3'
        }
    }
    output_type = 'string'

    def __init__(self):
        super().__init__()
        self.chroma_client = ChromaDBClient(
            persist_directory='./chroma_db',
            collection_name='test_collection',
            openai_api_base='http://localhost:9997/v1',
            openai_api_key='xxx',
            embedding_model='bge-m3'
        )

    def forward(self, question: str, k: int):
        search_results = self.chroma_client.vector_store.similarity_search(
            query=question,
            k=k
        )
        content = '\n---\n'.join([doc.page_content for doc in search_results])
        return content
