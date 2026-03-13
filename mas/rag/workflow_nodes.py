"""
工作流节点定义 - LangGraph工作流的各个处理步骤

包含6个主要节点：
1. preprocess - 请求预处理
2. local_rag_search - 本地RAG检索
3. evaluate_local_hit - 评估本地命中
4. allocate_local - 本地分配
5. request_remote - 远程请求
6. finalize - 最终化
"""

import logging
import time
from typing import Dict, Any
from .workflow_state import WorkflowState, AllocationScore

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """LangGraph工作流节点"""

    def __init__(self, rag_database):
        """
        初始化工作流节点

        Args:
            rag_database: LocalRAGDatabase 实例
        """
        self.rag = rag_database
        logger.info("✓ WorkflowNodes initialized")

    # ────────────────────────────────────────────────────────
    # 节点1：预处理
    # ────────────────────────────────────────────────────────

    async def preprocess(self, state: WorkflowState) -> WorkflowState:
        """
        节点1：请求预处理

        功能：
        - 验证任务请求格式
        - 初始化工作流元数据
        - 记录开始时间
        """
        try:
            import time
            state.start_time = time.time()

            logger.info(
                f"[{state.workflow_id}] Preprocessing request: "
                f"{state.task_request.get('task_type', 'unknown')}"
            )

            # 验证必要字段
            required_fields = ["task_type", "description"]
            for field in required_fields:
                if field not in state.task_request:
                    raise ValueError(f"Missing required field: {field}")

            logger.info(f"[{state.workflow_id}] ✓ Request validation passed")
            return state

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Preprocessing failed: {e}")
            state.error_message = str(e)
            return state

    # ────────────────────────────────────────────────────────
    # 节点2：本地RAG搜索
    # ────────────────────────────────────────────────────────

    async def local_rag_search(self, state: WorkflowState) -> WorkflowState:
        """
        节点2：本地RAG检索

        功能：
        - 使用Milvus检索相似任务
        - 检索该任务类型的最佳Agent
        - 补充完整元数据
        """
        try:
            start_time = time.time()

            if not state.task_embedding:
                logger.warning(f"[{state.workflow_id}] No embedding provided, skipping search")
                return state

            task_type = state.task_request.get("task_type")
            description = state.task_request.get("description", "")

            # 搜索相似任务
            logger.info(f"[{state.workflow_id}] Searching similar tasks...")
            state.local_search_results = await self.rag.search_tasks(
                query=description,
                task_type=task_type,
                top_k=5,
            )

            logger.info(
                f"[{state.workflow_id}] ✓ Found {len(state.local_search_results)} "
                f"similar tasks"
            )

            # 搜索最佳Agent
            logger.info(f"[{state.workflow_id}] Searching best agents...")
            state.best_agents = await self.rag.search_solutions(
                query=description,
                task_type=task_type,
                top_k=5,
            )

            logger.info(
                f"[{state.workflow_id}] ✓ Found {len(state.best_agents)} best agents"
            )

            # 记录执行时间
            elapsed = time.time() - start_time
            state.step_timings["local_rag_search"] = elapsed * 1000

            return state

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Local search failed: {e}")
            state.error_message = str(e)
            return state

    # ────────────────────────────────────────────────────────
    # 节点3：评估本地命中
    # ────────────────────────────────────────────────────────

    async def evaluate_local_hit(self, state: WorkflowState) -> str:
        """
        节点3：判断是否有足够好的本地方案

        Returns:
            "allocate_local" - 使用本地方案
            "request_remote" - 请求远程Agent
        """
        try:
            # 收集指标
            has_similar_task = len(state.local_search_results) > 0
            has_good_agent = any(
                agent.get("success_rate", 0) > 0.7 for agent in state.best_agents
            )

            # 计算置信度
            if has_similar_task:
                state.local_hit_confidence = min(
                    0.5 + len(state.local_search_results) * 0.1, 0.95
                )
            elif has_good_agent:
                state.local_hit_confidence = 0.8
            else:
                state.local_hit_confidence = 0.0

            logger.info(
                f"[{state.workflow_id}] Local hit confidence: {state.local_hit_confidence:.2%}"
            )

            # 决策逻辑
            if state.local_hit_confidence > 0.6:
                logger.info(f"[{state.workflow_id}] ✓ Using local resources")
                state.allocation_decision = "local_with_scoring"
                state.allocation_reason = (
                    f"Found {len(state.best_agents)} agents "
                    f"with confidence {state.local_hit_confidence:.2%}"
                )
                return "allocate_local"
            else:
                logger.info(
                    f"[{state.workflow_id}] ✗ No good local resources, "
                    f"requesting remote agents"
                )
                state.allocation_decision = "remote_fallback"
                state.allocation_reason = "Insufficient local resources"
                return "request_remote"

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Evaluation failed: {e}")
            return "request_remote"

    # ────────────────────────────────────────────────────────
    # 节点4：本地分配
    # ────────────────────────────────────────────────────────

    async def allocate_local(self, state: WorkflowState) -> WorkflowState:
        """
        节点4：使用本地方案分配

        功能：
        - 从最佳Agent中选择
        - 基于权重评分排序
        - 返回分配结果
        """
        try:
            start_time = time.time()

            if not state.best_agents:
                logger.warning(f"[{state.workflow_id}] No local agents available")
                return state

            # 获取权重用于评分
            weights = await self.rag.get_weights()

            logger.info(f"[{state.workflow_id}] Scoring agents with weights: {weights}")

            # 计算综合评分 (AEIC四层)
            scored_agents = []
            for agent in state.best_agents:
                agent_id = agent.get("agent_id")

                # 获取Agent信息
                agent_info = await self.rag.get_agent(agent_id)
                if not agent_info:
                    continue

                # 计算四层评分
                score = AllocationScore(
                    agent_id=agent_id,
                    ability=agent_info.get("success_rate", 0.5),
                    evidence=agent.get("success_rate", 0.5),
                    inference=1.0 - min(agent.get("similarity_distance", 1) / 10, 1.0),
                    conclusion=0.5,
                )

                # 计算总分
                score.calculate_total(weights)
                score.reasoning = {
                    "ability": agent_info.get("success_rate"),
                    "evidence": agent.get("success_rate"),
                    "inference": score.inference,
                    "distance": agent.get("similarity_distance"),
                }

                scored_agents.append(score)

            # 按评分排序
            scored_agents.sort(key=lambda x: x.total_score, reverse=True)

            # 选择最佳Agent
            state.selected_agents = [s.agent_id for s in scored_agents[:1]]
            state.agent_scores = {s.agent_id: s.total_score for s in scored_agents}

            logger.info(f"[{state.workflow_id}] ✓ Selected agents: {state.selected_agents}")

            # 生成分配结果
            state.allocation_result = {
                "type": "local",
                "selected_agents": state.selected_agents,
                "agent_scores": state.agent_scores,
                "scored_agents": [s.to_dict() for s in scored_agents],
                "confidence": state.local_hit_confidence,
                "timestamp": time.time(),
            }

            state.success = True

            # 记录执行时间
            elapsed = time.time() - start_time
            state.step_timings["allocate_local"] = elapsed * 1000

            return state

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Local allocation failed: {e}")
            state.error_message = str(e)
            return state

    # ────────────────────────────────────────────────────────
    # 节点5：远程请求
    # ────────────────────────────────────────────────────────

    async def request_remote(self, state: WorkflowState) -> WorkflowState:
        """
        节点5：请求远程Agent

        功能：
        - 发送广播请求到其他Agent
        - 收集响应
        - 评估最佳方案
        """
        try:
            start_time = time.time()

            task_type = state.task_request.get("task_type")
            description = state.task_request.get("description")

            logger.info(
                f"[{state.workflow_id}] Broadcasting request to remote agents "
                f"for task type: {task_type}"
            )

            # 这里应该实现跨Agent通信
            # 为了演示，模拟远程请求
            state.remote_responses = [
                {
                    "agent_id": 10,
                    "solution": f"Remote solution for {task_type}",
                    "success_rate": 0.9,
                    "timestamp": time.time(),
                }
            ]

            logger.info(
                f"[{state.workflow_id}] ✓ Received {len(state.remote_responses)} "
                f"remote solutions"
            )

            # 生成分配结果
            state.allocation_result = {
                "type": "remote",
                "remote_responses": state.remote_responses,
                "timestamp": time.time(),
            }

            state.selected_agents = [r["agent_id"] for r in state.remote_responses]
            state.success = True

            # 记录执行时间
            elapsed = time.time() - start_time
            state.step_timings["request_remote"] = elapsed * 1000

            return state

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Remote request failed: {e}")
            state.error_message = str(e)
            return state

    # ────────────────────────────────────────────────────────
    # 节点6：最终化
    # ────────────────────────────────────────────────────────

    async def finalize(self, state: WorkflowState) -> WorkflowState:
        """
        节点6：最终化处理

        功能：
        - 记录分配结果到RAG
        - 生成反馈用于权重学习
        - 返回最终结果
        """
        try:
            import uuid

            state.end_time = time.time()

            if not state.selected_agents:
                raise ValueError("No agents selected for allocation")

            record_id = str(uuid.uuid4())

            logger.info(
                f"[{state.workflow_id}] Recording allocation: {record_id}"
            )

            # 记录成功的分配
            await self.rag.record_success(
                record_id=record_id,
                task_id=state.task_request.get("task_id", "unknown"),
                agent_ids=state.selected_agents,
                feedback=f"Allocation completed for {state.allocation_decision}",
                success_score=0.8,
                metadata={
                    "workflow_id": state.workflow_id,
                    "allocation_decision": state.allocation_decision,
                    "agent_scores": state.agent_scores,
                    "confidence": state.local_hit_confidence,
                },
            )

            state.allocation_result["record_id"] = record_id

            logger.info(f"[{state.workflow_id}] ✓ Workflow completed successfully")

            return state

        except Exception as e:
            logger.error(f"[{state.workflow_id}] ✗ Finalization failed: {e}")
            state.error_message = str(e)
            state.success = False
            state.end_time = time.time()
            return state
