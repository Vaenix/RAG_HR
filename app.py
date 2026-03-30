import os
import shutil
import json
import re
from fastapi import UploadFile, File
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
        self.base_file_path = "datasets/模拟公司员工手册.md"
        self.is_base_mode = True
        self.current_identity = "创新科技股份有限公司的HR助手"
        self.current_doc_name = os.path.basename(self.base_file_path)
        self.ui_config = {}
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
        self.refresh_runtime_profile(self.base_file_path)
        self.full_chain = self.build_full_chain()
        print("✅ 创新科技HR助手RAG系统初始化完成!")
    def load_markdown_content(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    def extract_heading_topics(self, md_content, limit=6):
        headings = []
        seen = set()
        for line in md_content.splitlines():
            match = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", line)
            if not match:
                continue
            heading = re.sub(r"[`*_>\[\]()]", "", match.group(1)).strip()
            if not heading or heading in seen:
                continue
            seen.add(heading)
            headings.append(heading)
            if len(headings) >= limit:
                break
        return headings
    def build_fallback_ui_config(self, md_content, file_name):
        topics = self.extract_heading_topics(md_content)
        identity = self.current_identity or "智能知识助手"
        short_topics = topics[:4] if topics else ["核心制度说明", "流程规范", "常见问题", "重点要求"]
        capability_items = [
            {"title": topic[:12], "description": f"快速查看与“{topic[:12]}”相关的规则与说明"}
            for topic in short_topics
        ]
        example_questions = [
            f"请介绍一下{topic}" for topic in (topics[:4] or ["当前文档的核心内容", "文档中的重点流程", "文档里的常见要求", "文档适用范围"])
        ]
        greeting_topics = "\n".join([f"- **{topic}**" for topic in short_topics])
        return {
            "app_title": identity[:18],
            "app_subtitle": f"基于《{file_name}》的实时知识问答",
            "welcome_title": f"欢迎使用{identity}",
            "welcome_description": f"当前页面文案已绑定到《{file_name}》，上传新的 Markdown 后会自动切换为对应知识库内容。",
            "chat_title": f"与{identity}对话",
            "chat_description": f"当前知识库：{file_name}",
            "assistant_greeting": (
                f"您好，我是{identity}。\n\n"
                f"当前已加载《{file_name}》，您可以优先咨询这些方向：\n"
                f"{greeting_topics}\n\n"
                "如果您需要，我也可以根据当前文档继续追问、解释规则或梳理流程。"
            ),
            "assistant_tagline": "页面文案会随知识库实时切换",
            "placeholder": "请输入与当前知识库相关的问题，Shift + Enter 可换行",
            "upload_label": "上传并切换知识库",
            "example_questions": example_questions,
            "capability_items": capability_items
        }
    def generate_ui_config(self, md_content, file_name):
        fallback = self.build_fallback_ui_config(md_content, file_name)
        prompt = f"""
你是一个企业知识库产品的中文前端内容策划。请根据给定 Markdown 文档内容，为问答页面生成一份简洁、专业、适合直接展示的 UI 文案配置。

要求：
1. 只输出 JSON，不要添加解释。
2. 所有文案必须基于当前文档主题，不能写死“HR”或特定公司名，除非文档确实如此。
3. 文案风格简洁、专业、自然。
4. 标题不要太长，适合网页展示。
5. `example_questions` 返回 4 个问题，`capability_items` 返回 4 个对象，每个对象包含 `title` 和 `description`。
6. `assistant_greeting` 使用 Markdown，适合放在聊天首条消息中。

JSON 字段如下：
{{
  "app_title": "顶部主标题",
  "app_subtitle": "顶部副标题",
  "welcome_title": "欢迎区标题",
  "welcome_description": "欢迎区说明",
  "chat_title": "聊天区标题",
  "chat_description": "聊天区描述",
  "assistant_greeting": "首条欢迎消息",
  "assistant_tagline": "助手状态短句",
  "placeholder": "输入框占位文案",
  "upload_label": "上传按钮文案",
  "example_questions": ["问题1", "问题2", "问题3", "问题4"],
  "capability_items": [
    {{"title": "能力1", "description": "描述1"}},
    {{"title": "能力2", "description": "描述2"}},
    {{"title": "能力3", "description": "描述3"}},
    {{"title": "能力4", "description": "描述4"}}
  ]
}}

文档文件名：{file_name}
文档内容（节选）：
{md_content[:3500]}
"""
        try:
            response = self.model.invoke(prompt)
            raw_text = response.content.strip()
            matched = re.search(r"\{[\s\S]*\}", raw_text)
            if not matched:
                return fallback
            parsed = json.loads(matched.group(0))
            parsed["example_questions"] = (parsed.get("example_questions") or fallback["example_questions"])[:4]
            parsed["capability_items"] = (parsed.get("capability_items") or fallback["capability_items"])[:4]
            for key, value in fallback.items():
                if not parsed.get(key):
                    parsed[key] = value
            return parsed
        except Exception as e:
            print(f"⚠️ 生成动态UI配置失败，已回退默认文案: {e}")
            return fallback
    def infer_identity(self, md_content):
        summary_prompt = (
            "请简要描述这个文档对应的助手身份和所属主题。"
            "只输出一句中文身份描述，不要超过20个字。"
        )
        try:
            summary_res = self.model.invoke(f"{summary_prompt}\n\n文档内容片段：{md_content[:700]}")
            identity = summary_res.content.strip()
            identity = re.sub(r"[\n\r]+", " ", identity)
            return identity[:20] if identity else "智能知识助手"
        except Exception as e:
            print(f"⚠️ 推断助手身份失败，已使用默认身份: {e}")
            return "智能知识助手"
    def refresh_runtime_profile(self, file_path):
        md_content = self.load_markdown_content(file_path)
        self.current_doc_name = os.path.basename(file_path)
        self.current_identity = self.infer_identity(md_content)
        self.ui_config = self.generate_ui_config(md_content, self.current_doc_name)
        self.ui_config["identity"] = self.current_identity
        self.ui_config["doc_name"] = self.current_doc_name
    def rebuild_index(self, file_path):
        """核心：根据用户上传的文件实时重建索引"""
        print(f"🔄 正在根据新文件重建索引: {file_path}")
        self.is_base_mode = False
        md_content = self.load_markdown_content(file_path)

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
        
        # 2. 覆盖当前的 retriever
        vector_store = FAISS.from_documents(final_splits, embedding=self.embed)
        self.retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        self.refresh_runtime_profile(file_path)
        self.full_chain = self.build_full_chain()
        print("✅ 动态知识库切换成功！")
    
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
        file_path = self.base_file_path
        md_content = self.load_markdown_content(file_path)
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
        你是{identity}。你将回答有关公司的员工福利、薪酬结构、
        工作时间、出勤政策、假期休假、绩效评估、培训发展、纪律惩戒、健康安全、信息安全、
        多样性包容、环境可持续发展、离职程序以及其他与我司HR相关的话题。
        
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
            input_variables=["chat_history", "context", "question","identity"],
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
                "question": itemgetter("question"),
                "identity": lambda x: self.current_identity
            }
            |
            {
                "context": itemgetter("relevant_docs") | RunnableLambda(self.format_context),
                "chat_history": itemgetter("chat_history"), 
                "question": itemgetter("question"),
                "identity": lambda x: self.current_identity
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
        if self.is_base_mode:
            # 基础模式：保留你的“安检 + 分流”逻辑
            branch_node = RunnableBranch(
                (lambda x: "是" in x["question_is_relevant"].lower(), relevant_question_chain),
                (lambda x: "否" in x["question_is_relevant"].lower(), irrelevant_question_chain),
                irrelevant_question_chain
            )
            full_chain = (
                {
                    "question_is_relevant": guardrail_chain,
                    "messages": RunnablePassthrough()
                }
                | branch_node
            )
        else:
            # 上传模式：跳过安检，直接进入 RAG 流程
            full_chain = (
                {
                    "question": itemgetter("messages") | RunnableLambda(self.extract_question),
                    "chat_history": itemgetter("messages") | RunnableLambda(self.extract_history) | RunnableLambda(self.format_chat_history)
                }
                | relevant_question_chain
                | RunnableLambda(lambda x: {**x, "question_is_relevant": "是"}) # 伪造一个“是”，防止后端 chat 函数报错
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
    def get_ui_payload(self):
        return {
            "mode": "base" if self.is_base_mode else "uploaded",
            "identity": self.current_identity,
            "doc_name": self.current_doc_name,
            **self.ui_config
        }

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
            "company": rag_system.current_identity,
            "doc_name": rag_system.current_doc_name
        }
    except Exception as e:
        return {"error": f"获取统计信息失败: {str(e)}"}
@app.get("/api/ui-config")
async def get_ui_config():
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    return rag_system.get_ui_payload()
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1. 创建临时存放目录
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    # 2. 保存文件到本地
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. 调用 RAG 系统的重建方法
        rag_system.rebuild_index(file_path)
        return {
            "status": "success",
            "filename": file.filename,
            "ui_config": rag_system.get_ui_payload()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)
