# 固定状态与动作的核查诊断（2026-10-29）

研究目标保持“预算约束下的多智能体博弈共识及影视推荐应用”。本补充实现用于解释核查行动能否利用已知机会，不将研究改为负结果复盘。

## 与当前运行批次的关系

当前paper_v2r1和decision_linked_probe_v3的输入、源码和结果保持原样。本入口在experiments/bound_opportunity_probe.py，独立结果目录bound_opportunity_v3；不覆盖或解除旧release门。不得在运行中替换core.py、decision_review.py或repair_probe.py。新入口依赖源码逐文件锁定，后续源码变动须新建root；已完成分支复用，未知请求由现有耐久传输纪律保留、不自动补发。

## 修正内容与依据

1. 固定状态：直接使用manifest中的state，不调用initialize，不重新生成初始答案。
2. 固定动作：manifest明确target_id、A/B来源ID；每个普通修订与受限核查分支得到相同状态、同样已暴露材料和报告。不存在按哈希另选材料。
3. 配对分支：每题empty、A、B、AB × ordinary/linked，最多8请求。空材料分支控制重新思考的收益；A/B/AB识别单份价值与交互。
4. 证据归因：无引文=none；仅原有=initial；仅指派材料（允许指派集合的子集）=assigned；二者同时引用=joint。分类依据引文来源，而不是“是否引用完全部指派材料”。这些类别记录引用行为，不能证明语义依赖或因果贡献。
5. 执行路径：新invoke只调用适配器的放宽校验一次，再提交。旧core.invoke先严格validate_report、后linked_validate的顺序会提前拒绝初始证据修订；本入口不经过该路径。保留旧运行源码，避免变更冻结实验。
6. 门槛分离：统计合同有效、AB纠错率、正确维持率、误改、empty收益、相对empty的材料收益、联合独有收益。最低诊断门要求全部合同有效、有维持对照且不误改、观察到材料净增量与联合信号；不再只写improved>0。

## 已准备的五类构造对照

默认任务是封闭记录域中的明确事实，实体使用虚构编号；标签由显式记录规则确定，不依赖外部知识。任务ID不暴露对照类别。

- 单份纠错：初始错误；A明确说明目标事实，B关于另一实体。
- 联合纠错：A给出实体类别，B给出类别规则；AB才能在所给记录下建立目标事实。
- 正确维持：初始正确且已见充分证据；继续核查应保留决策。
- 证据不足：只有另一实体的记录；正确答案为not_enough_info，不能强行修改。
- 重复资料：A/B内容相同；用来观察重复不应冒充互补。

这些是执行能力与信息使用诊断，不能作为自然问答/影视推荐主表、自然联合推理或博弈优势的结果。单份和联合支撑由记录构造保证；模型能否利用由真实调用决定。

## 自然任务清单格式

可用 --manifest 传入JSON列表，每项包含：

```
{
 "task": "完整原生任务对象，不含evaluation标签",
 "state": "已封存训练案例中的baseline完整对象",
 "reports": "同一案例initial_events中有效报告，保留原顺序",
 "target_id": "baseline中的活跃主张ID",
 "sources": {"A": ["指定来源ID"], "B": ["指定来源ID"]},
 "stratum": "single / joint / keep / insufficient / redundant",
 "adjudication": {
   "status": "supported", "author": "裁决者和方式",
   "reason": "全文证据到结论的桥梁及必要前提",
   "span_ids": ["对应绑定材料的不可变span ID"]
 },
 "evaluation": "仅评估代码使用的原生标签对象"
}
```

自然机会选题仅接受fit/calibration分区；supported必须有绑定来源的原文span及裁决记录。单纯gold位于Top10外只证明机械换入空间，不能替代语义支持裁决。QA裁决用全文，分明确支持/需要额外前提/不支持/无法判断；后三类不能标supported。代理裁决必须标明身份，不冒充人工双人标注。更强语义裁决不能由结构校验自动保证。

清单及evaluation私有留在本地；请求只由task、state、reports和对应action构造，不发送stratum、裁决或evaluation。构造诊断标签本来已知，不将其宣称盲选自然测试。受控恢复使用手工固定错误状态，真实自然评价使用原始未干预状态，两者分表。

## 收益与预算启动检查

对同一题同一方法，记封存后质量为Q_empty,Q_A,Q_B,Q_AB。单份收益分别为Q_A-Q_empty、Q_B-Q_empty，联合收益为Q_AB-Q_empty，交互为Q_AB-Q_A-Q_B+Q_empty。仅四分支全部有效的题用于这些差值；无效分支仍进入合同率，并保留回退结果，不能当作有用动作信号。

budget_check要求存在有效的正材料收益、动作价值差异、候选动作数大于正预算。预算0是控制。该检查只判断是否值得进行预算比较，不认证贡献估计器，也不把构造任务收益当成自然域收益。正式比较仍须相同信息、动作权限、调用/材料预算下比较固定、随机/贪心、中央、同权限单Agent及博弈；中央精确最优在同一目标下作为参照，不预设博弈必须胜过它。

## 运行

```bash
bash scripts/run_bound_opportunity_probe.sh --prepare
bash scripts/run_bound_opportunity_probe.sh --run
bash scripts/run_bound_opportunity_probe.sh --report
bash scripts/run_bound_opportunity_probe.sh --budget-check
```

默认prepare不联网。run为qwen_flash（现有廉价配置、关闭思考），5题最多40请求，硬上限0.2元。全局沿用3秒节流，当前Claude批次也使用同一节流锁；建议已有批次结束后运行，避免竞争吞吐。运行可中断续跑同一命令；不改已冻结清单/费用上限。封存后才汇总评价。--manifest自定义清单需同时指定新的--root，prepare/run/report使用相同参数。

## 旧stress的准确用途

现有prepare_protocol_v3_stress.py只删除未读来源、复制来源、增加低词面重叠资料，没有干预初始决策，也不保证移除的是决定性证据。因此旧stress仅作为资料扰动稳健性评价，不写成已知错误恢复实验。不删除结果、不强制停下他人运行。本文的新构造诊断补充“固定错误状态+保留充分纠正证据”的恢复实验，独立记账和解释。

## 推进顺序

完成当前主批封存与九方法比较；新诊断单独运行；若存在材料增量和行动差异，使用训练侧经过裁决的自然任务复核，再重采贡献标签、检验预计选择和中央/博弈比较。若只有构造域恢复，优先改进自然检索/比较材料的证据供给，不继续盲目扩大零收益训练数据。诊断结果回填主线方法的设计依据，研究目标保持不变。

最初prepare的bound_opportunity_v1仅为未运行的开发快照（0请求），实体ID去除类别提示后按不可变纪律改用bound_opportunity_v3；不用旧目录启动。
