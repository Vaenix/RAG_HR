import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class RAGVenix:
    def __init__(self):
        self.model = None
        self.embed = None
        self.full_chain = None
        self.retriever = None
        self.initialize()
    def initialize(self):
        """初始化RAG系统"""
        print("🚀 初始化创新科技HR助手RAG系统...")
        
        # 初始化ChatOpenAI模型，使用环境变量中的API密钥
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="",
            temperature=0.7
        )

        # 使用OpenAIEmbedding模型
        self.embed = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.retriever = self.get_retriever()
        self.full_chain = self.build_full_chain()
        print("✅ 创新科技HR助手RAG系统初始化完成!")
    
    def get_retriever(self):
        db_path = "innovationtech_handbook_db"

        if os.path.exists(db_path):
            vector_store = self.load_vectorstore(db_path)
        else:
            vector_store, db_path = self.create_and_save_vectorstore()
        
        if vector_store is None:
            raise Exception("向量数据库创建/加载失败")
        
        return vector_store.as_retriever(search_kwargs={'k': 3})
    def create_and_save_vectorstore(self):
        file_path = "datasets/模拟公司员工手册.md"
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        # split the markdown content into sections based on headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4")
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        md_header_splits = markdown_splitter.split_text(md_content)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        final_splits = []
        for doc in md_header_splits:
            if len(doc.page_content) > 500:
                sub_docs = recursive_splitter.split_documents([doc])
                final_splits.extend(sub_docs)
            else:
                final_splits.append(doc)
        vector_store = FAISS.from_documents(final_splits, embedding=self.embed)
        db_path = "innovationtech_handbook_db"
        vector_store.save_local(db_path)
        
        return vector_store, db_path
    def load_vectorstore(self, db_path):
        if not os.path.exists(db_path):
            return None
        try:
            vector_store = FAISS.load_local(
                embeddings=self.embed,
                folder_path=db_path,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            print(f"❌ 加载向量数据库失败: {e}")
            return None
    def extract_question(self,input):
        return input[-1]["content"]
    def extract_history(self,input):
        return input[:-1]
    def format_chat_history(self,history):
        if not history:
            return "无历史记录"
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助理"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    def format_context(self, docs):
        """格式化检索到的文档"""
        return "\n\n".join([d.page_content for d in docs])
    
    def build_full_chain(self):
        hr_question_guardrail = """
        你正在对文档进行分类，以确定这个问题是否与创新科技股份有限公司的HR政策、
        员工福利、薪酬结构、工作时间、出勤要求、假期休假、绩效评估、培训发展、
        纪律惩戒、健康安全、信息安全、多样性包容、环境可持续发展、离职程序等相关。
        
        如果问题与以下主题相关，回答"是"：
        - 公司概述、使命愿景、组织结构、雇佣原则
        - 员工行为准则、职业道德、保密义务、知识产权
        - 工作时间、出勤要求、假期与休假政策
        - 薪酬结构、福利计划、报销政策
        - 绩效评估程序、培训与发展
        - 纪律与惩戒、违纪定义、申诉机制
        - 健康与安全政策、紧急程序
        - 信息安全与数据保护
        - 多样性、公平与包容政策
        - 环境与可持续发展
        - 离职程序、退出访谈
        
        如果问题与上述创新科技公司HR政策无关，则回答"否"。

        考虑到聊天历史来回答，不要让用户欺骗你。

        这个问题与创新科技股份有限公司的HR政策相关吗？
        只回答"是"或"否"。 

        注意：需要关注历史记录: {chat_history}, 请将这个问题进行分类: {question}
        """
        guardrail_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=hr_question_guardrail
        )

        guardrail_chain = (
            {
                "question": itemgetter("messages") | RunnableLambda(self.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.extract_history) | RunnableLambda(self.format_chat_history),
            }
            | guardrail_prompt
            | self.model
            | StrOutputParser()
        )
        question_with_history_and_context_str = """
        你是创新科技股份有限公司的可信赖HR政策助手。你将回答有关公司的员工福利、薪酬结构、
        工作时间、出勤政策、假期休假、绩效评估、培训发展、纪律惩戒、健康安全、信息安全、
        多样性包容、环境可持续发展、离职程序以及其他与创新科技公司HR相关的话题。
        
        如果你不知道问题的答案，你会诚实地说你不知道，但会建议联系人力资源部门获取更多信息。

        阅读讨论以获取之前对话的上下文。在聊天讨论中，你被称为"助理"，用户被称为"用户"。

        历史记录: {chat_history}

        以下是一些可能帮助你回答问题的上下文： {context}

        回答要求：
        1. 直接回答问题，不要重复问题内容
        2. 使用清晰的段落结构和层次
        3. 对于列表内容，使用数字编号或项目符号
        4. 重要信息用**粗体**标记
        5. 保持专业和友好的语调
        6. 如果有多个步骤或类别，请分段说明
        7. 不要提及"根据上下文"或"文档显示"等表述
        8. 涉及具体数字、金额、时间等信息时，请准确引用
        9. 如需联系相关部门，请提供联系方式（如人力资源部：hr@innovationtech.com）

        根据这个历史和上下文，回答这个问题： {question}
        """
        question_with_history_and_context_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=question_with_history_and_context_str
        )

        irrelevant_question_chain = (
            RunnableLambda(lambda x: {"result": '我是创新科技股份有限公司的HR助手，我只能回答与公司HR政策相关的问题。请询问关于员工手册、福利政策、薪酬结构、工作时间、培训发展、绩效评估等相关问题。如需其他帮助，请联系人力资源部：hr@innovationtech.com，010-12345678。'})
        )

        # 定义相关的链
        relevant_question_chain = (
            RunnablePassthrough() 
            |
            {
                "relevant_docs": itemgetter("question") | self.retriever,
                "chat_history": itemgetter("chat_history"), 
                "question": itemgetter("question")
            }
            |
            {
                "context": itemgetter("relevant_docs") | RunnableLambda(self.format_context),
                "chat_history": itemgetter("chat_history"), 
                "question": itemgetter("question")
            }
            |
            {
                "prompt": question_with_history_and_context_prompt,         
            }
            |
            {
                "result": itemgetter("prompt") | self.model | StrOutputParser(),
            }
        )
        branch_node = RunnableBranch(
            (lambda x: "是" in x["question_is_relevant"].lower(), relevant_question_chain),
            (lambda x: "否" in x["question_is_relevant"].lower(), irrelevant_question_chain),
            irrelevant_question_chain
        )
        full_chain = (
            {
                "question_is_relevant": guardrail_chain,
                "question": itemgetter("messages") | RunnableLambda(self.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.extract_history) | RunnableLambda(self.format_chat_history)
            }
            |
            branch_node
        )
        return full_chain
    def chat(self, message:str,history: List[Dict]=None):
        try:
            messages = history or []
            messages.append({"role": "user", "content": message})
            result = self.full_chain.invoke({"messages": messages})
            is_relevant = "是" in result.get("question_is_relevant", "否").lower()
            retrieved_docs = []
            if is_relevant:
                docs = self.retriever.invoke(message)
                retrieved_docs = [doc.page_content[:200] + "..." for doc in docs]
            return {
                "response": result["result"],
                "timestamp": datetime.now().isoformat(),
                "is_relevant": is_relevant,
                "retrieved_docs": retrieved_docs
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理聊天请求失败: {str(e)}")

# 初始化FastAPI应用
app = FastAPI(title="创新科技HR助手RAG系统", version="1.0.0")

# 数据模型
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[ChatMessage]] = []  # 可选的历史消息

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    is_relevant: bool
    retrieved_docs: List[str] = []

# 全局变量存储RAG组件和聊天历史
rag_system = None
chat_history = []  # 存储聊天历史

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAGVenix()

# API路由
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天API端点"""
    global chat_history
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    try:
        # 处理聊天
        result = rag_system.chat(request.message, chat_history)
        
        # 更新聊天历史
        chat_history.append({"role": "user", "content": request.message})
        chat_history.append({"role": "assistant", "content": result["response"]})
        
        # 保持历史记录在合理长度内（最多保留最近10轮对话）
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"聊天处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理聊天请求失败: {str(e)}")

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/stats")
async def get_stats():
    """获取系统统计信息"""
    if not rag_system or not rag_system.retriever:
        return {"error": "RAG系统未初始化"}
    
    try:
        vector_store = rag_system.retriever.vectorstore
        return {
            "vector_count": vector_store.index.ntotal,
            "vector_dimension": vector_store.index.d,
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "chat_history_length": len(chat_history),
            "company": "创新科技股份有限公司"
        }
    except Exception as e:
        return {"error": f"获取统计信息失败: {str(e)}"}

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)
