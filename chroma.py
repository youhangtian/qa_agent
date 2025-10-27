import os 
import chromadb

from docx import Document as DocxDocument
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

class ChromaDBClient:
    def __init__(self,
                 persist_directory: str = './chroma_db',
                 collection_name: str = 'test_collection',
                 openai_api_base: str = 'http://localhost:9997/v1',
                 openai_api_key: str = 'xxx',
                 embedding_model: str = 'bge-m3'):
        headers_to_split_on = [
            ('#', 'Header 1'),
            ('##', 'Header 2'),
            ('###', 'Header 3'),
            ('####', 'Header 4'),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

        self.embeddings = OpenAIEmbeddings(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model=embedding_model,
        )

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )


    def add_data(self, file_name):
        name = file_name.split('/')[-1].split('.')[0]

        doc = DocxDocument(file_name)
        texts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip() 
            style = paragraph.style.name

            if not text:
                continue 

            if style.startswith('Heading'):
                try:
                    count = int(style.split('Heading')[-1])
                except:
                    count = 4 

                text = ''.join(['#' for _ in range(count)]) + ' ' + text
            texts.append(text)
        doc_text = '\n'.join(texts)

        splits = self.markdown_splitter.split_text(doc_text)
        for doc in splits:
            headers = '\n'.join([val for key, val in doc.metadata.items() if key.startswith('Header')])
            doc.page_content = headers + '\n' + doc.page_content
            doc.metadata['source'] = file_name
        ids = [f"{name}_{i}" for i in range(len(splits))]

        self.vector_store.add_documents(splits, ids=ids)
        print(f'{file_name} added, {len(splits)} chunks, total {self.collection.count()} chunks')
        print('--------------------------------')

    def list_datas(self):
        count = self.collection.count()
        datas = self.collection.peek(count)

        ret = []
        for ids, metadatas in zip(datas['ids'], datas['metadatas']):
            ret.append((ids, metadatas))
        return ret

    def search(self,
               query: str,
               k: int = 3):
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )
        return results
    
    def delete(self, file_name):
        count = self.collection.count()
        datas = self.collection.peek(count)

        ids = []
        for id, metadata in zip(datas['ids'], datas['metadatas']):
            if file_name in metadata.get('source', ''):
                ids.append(id)

        if ids:
            self.collection.delete(ids=ids)

        return ids

if __name__ == '__main__':
    client = ChromaDBClient()
    
    # for file_name in os.listdir('./data'):
    #     if not file_name.endswith('.docx'):
    #         continue 
    #     print('Processing file:', file_name)
    #     client.add_data(os.path.join('./data', file_name))

    datas = client.list_datas()
    print('datas len:', len(datas))
    for data in datas:
        print(data)
    print('--------------------------------')

    file_name = '七天综合绍兴城市大脑运行日志20250615.docx'
    ids = client.delete(file_name)
    print('Deleted ids:', ids)

    datas = client.list_datas()
    print('datas len:', len(datas))
    for data in datas:
        print(data) 
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    print('Done')