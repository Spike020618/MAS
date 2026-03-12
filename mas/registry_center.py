"""
Registry Center - 分布式多智能体系统的基础设施

职责：
1. 服务发现（Agent注册/查询）
2. 任务公告板（发布/查询任务）
3. 历史存储（共识记录）

不负责：
- 任务调度决策
- Agent间通信路由
- 共识仲裁
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse
import time
from typing import Dict, List, Optional
import json

app = FastAPI(title="Multi-Agent Registry Center")

# 允许跨域（用于分布式场景）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 数据模型 =================

class AgentInfo(BaseModel):
    host: str
    port: int
    model: str
    role: str
    capabilities: Optional[Dict] = {}

class TaskInfo(BaseModel):
    description: str
    initiator: int
    requirements: Optional[Dict] = {}

class ConsensusRecord(BaseModel):
    task_id: int
    initiator: int
    result: Dict
    participants: List[int]
    utility: float
    rounds: int

# ================= 全局状态 =================

# Agent注册表
agent_registry: Dict[str, Dict] = {}

# 任务池
task_pool: List[Dict] = []

# 共识历史
consensus_history: List[Dict] = []

# 统计信息
stats = {
    "total_agents_joined": 0,
    "total_tasks_published": 0,
    "total_consensus_reached": 0,
    "start_time": time.time()
}

# ================= API端点 =================

@app.post("/register")
async def register_agent(info: AgentInfo):
    """Agent加入网络时注册"""
    agent_id = f"{info.host}:{info.port}"
    
    agent_registry[agent_id] = {
        "model": info.model,
        "role": info.role,
        "capabilities": info.capabilities,
        "status": "online",
        "registered_at": time.time(),
        "last_seen": time.time()
    }
    
    stats["total_agents_joined"] += 1
    
    print(f"\n{'='*60}")
    print(f"✓ 新Agent加入网络")
    print(f"  ID: {agent_id}")
    print(f"  模型: {info.model}")
    print(f"  角色: {info.role}")
    print(f"  能力: {list(info.capabilities.keys())}")
    print(f"  当前网络规模: {len(agent_registry)} 个Agent在线")
    print(f"{'='*60}\n")
    
    return {
        "status": "success",
        "message": f"已加入网络，当前有 {len(agent_registry)} 个Agent在线",
        "agent_id": agent_id,
        "network": list(agent_registry.keys())
    }

@app.post("/unregister/{port}")
async def unregister_agent(port: int):
    """Agent退出网络"""
    agent_id = None
    for aid in list(agent_registry.keys()):
        if str(port) in aid:
            agent_id = aid
            break
    
    if agent_id:
        del agent_registry[agent_id]
        print(f"⚠️  Agent {agent_id} 已退出网络")
        return {"status": "success", "message": "已退出网络"}
    else:
        raise HTTPException(status_code=404, detail="Agent not found")

@app.get("/discover")
async def discover_agents(role: Optional[str] = None):
    """服务发现：查询可用的Agent"""
    if role:
        agents = {k: v for k, v in agent_registry.items() if v['role'] == role}
    else:
        agents = agent_registry
    
    return {
        "total": len(agents),
        "agents": agents,
        "timestamp": time.time()
    }

@app.post("/publish_task")
async def publish_task(task: TaskInfo):
    """发布任务到任务池（Registry只是公告板）"""
    task_id = len(task_pool)
    
    task_record = {
        "id": task_id,
        "description": task.description,
        "initiator": task.initiator,
        "requirements": task.requirements,
        "status": "open",
        "published_at": time.time(),
        "responses": []
    }
    
    task_pool.append(task_record)
    stats["total_tasks_published"] += 1
    
    print(f"\n{'='*60}")
    print(f"📢 新任务发布")
    print(f"  任务ID: {task_id}")
    print(f"  描述: {task.description}")
    print(f"  发起者: Agent {task.initiator}")
    print(f"  需求: {task.requirements}")
    print(f"{'='*60}\n")
    
    return {
        "status": "success",
        "task_id": task_id,
        "message": "任务已发布到公告板"
    }

@app.get("/tasks")
async def get_tasks(status: Optional[str] = None):
    """查询任务列表"""
    if status:
        tasks = [t for t in task_pool if t['status'] == status]
    else:
        tasks = task_pool
    
    return {
        "total": len(tasks),
        "tasks": tasks
    }

@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """查询特定任务"""
    if task_id >= len(task_pool):
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_pool[task_id]

@app.post("/store_consensus")
async def store_consensus(record: ConsensusRecord):
    """存储共识结果（只存储，不验证）"""
    consensus_record = {
        "task_id": record.task_id,
        "initiator": record.initiator,
        "result": record.result,
        "participants": record.participants,
        "utility": record.utility,
        "rounds": record.rounds,
        "timestamp": time.time()
    }
    
    consensus_history.append(consensus_record)
    stats["total_consensus_reached"] += 1
    
    # 更新任务状态
    if record.task_id < len(task_pool):
        task_pool[record.task_id]['status'] = 'completed'
    
    print(f"\n{'='*60}")
    print(f"✓ 共识记录已存储")
    print(f"  任务ID: {record.task_id}")
    print(f"  收益U: {record.utility:.2f}")
    print(f"  轮次: {record.rounds}")
    print(f"  参与者: {len(record.participants)} 个Agent")
    print(f"{'='*60}\n")
    
    return {
        "status": "stored",
        "total_records": len(consensus_history)
    }

@app.get("/consensus_history")
async def get_consensus_history(limit: int = 10):
    """查询共识历史"""
    return {
        "total": len(consensus_history),
        "history": consensus_history[-limit:]
    }

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    uptime = time.time() - stats["start_time"]
    
    return {
        "agents_online": len(agent_registry),
        "total_agents_joined": stats["total_agents_joined"],
        "total_tasks_published": stats["total_tasks_published"],
        "total_consensus_reached": stats["total_consensus_reached"],
        "tasks_pending": len([t for t in task_pool if t['status'] == 'open']),
        "uptime_seconds": uptime,
        "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Registry Center",
        "agents_online": len(agent_registry),
        "tasks_pending": len([t for t in task_pool if t['status'] == 'open'])
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Multi-Agent Registry Center",
        "version": "1.0.0",
        "description": "分布式多智能体系统的基础设施",
        "endpoints": [
            "POST /register - Agent注册",
            "GET /discover - 服务发现",
            "POST /publish_task - 发布任务",
            "GET /tasks - 查询任务",
            "POST /store_consensus - 存储共识",
            "GET /stats - 统计信息",
            "GET /health - 健康检查"
        ]
    }

# ================= 启动服务 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registry Center for Multi-Agent System")
    parser.add_argument("--port", type=int, default=9000, help="Registry端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌐 Registry Center 启动")
    print("="*70)
    print(f"   端口: {args.port}")
    print(f"   地址: http://{args.host}:{args.port}")
    print("   职责: 服务发现 + 任务公告板 + 历史存储")
    print("   注意: 此服务不做调度决策，仅提供基础设施")
    print("="*70)
    print("\n📋 等待Agent加入网络...\n")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )
