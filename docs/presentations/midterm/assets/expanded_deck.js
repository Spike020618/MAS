const q=t=>[...document.querySelectorAll('.s')].find(s=>s.querySelector('h2')?.textContent.trim()===t);
const layout=(chapter,title,sub,img,notes)=>`<div class="head"><span>${chapter}</span><span></span></div><h2>${title}</h2><p class="sub">${sub}</p><div class="visual-layout"><div class="visual"><img src="${img}" alt="${title}"></div><div class="explain">${notes.map(x=>`<article><h3>${x[0]}</h3><p>${x[1]}</p></article>`).join('')}</div></div>`;
const add=(after,chapter,title,sub,img,notes)=>{const s=document.createElement('section');s.className='s';s.innerHTML=layout(chapter,title,sub,img,notes);after.after(s);return s};

q('用公开机制调控信息生产').innerHTML=layout('研究内容及进展｜研究框架','AEIC 机制的研究框架','公开合同连接个体行动、多轮协作与集体决策','assets/fig_research_framework.png',[
['输入','任务、公开合同、Agent 配置和信息可见规则共同构成决策环境。'],['作用路径','合同先影响行动，行动再选择真实资源路径并生成 AEIC 报告。'],['输出','系统同时记录行为、集体准确率和资源成本。']]);
q('AEIC 将推理过程转化为可结算信号').innerHTML=layout('研究内容及进展｜关键技术 1','AEIC 结构化报告接口','四个字段提供可解析的语义对象，验证器据此产生信用信号','assets/fig_aeic_overview.png',[
['报告内容','A 记录前提，E 记录证据，I 记录推理，C 记录最终立场。'],['可观测信号','完整性、一致性、新颖证据、核验证据、有用纠正和无支持翻转。'],['结算作用','报告信号进入信用计算、成本扣除和信誉更新。']]);
q('验证、信用和信誉形成闭环').innerHTML=layout('研究内容及进展｜关键技术 3','三轮同步审议与结算时序','合同在生成前公开，行动在回答前声明，标签只在最终结算阶段使用','assets/fig_runtime_sequence.png',[
['Commit','Leader 先公开合同，避免根据回答事后改变规则。'],['Choose 与 Execute','Follower 先选行动，再执行对应的 Prompt、Token 和验证路径。'],['Settle','每轮保存报告、信用、成本与信誉，第三轮显式结论进入多数决策。']]);
q('同一问题只改变公开规则').innerHTML=layout('研究内容及进展｜实验设计','配对实验与证据层级','同一道题保持模型、Seed、五个 Agent、三轮审议和信息策略一致，只改变公开合同','assets/fig_experimental_design.png',[
['确认性比较','No Mechanism、Outcome Only 与 Static AEIC 构成基线、一般奖励对照和机制处理组。'],['主要结果','观察高资源行动率、第三轮集体准确率和 Token 成本。'],['证据规模','八组冻结实验共 5,850 条问题级记录，主实验包含 450 道配对题。']]);
let p=q('行动随奖励和成本呈方向一致的变化');
p=add(p,'研究内容及进展｜实验四：信息环境','不同信息环境下的机制效果','改变 Agent 可见的证据、社会信息、复制压力和反证信息','assets/fig3_information_conditions.png',[
['决策质量','四种环境下 Static AEIC 均高于对应的无机制条件。'],['行动响应','高资源行动率在不同环境中保持明显提升，复制压力环境响应最高。'],['结论','主效应不依赖单一信息结构，但效应大小随环境变化。']]);
p=add(p,'研究内容及进展｜实验五：消融与规模','组件扰动与 Panel 规模','信号、历史策略和 Agent 数量变化用于检查结果稳定性','assets/fig4_ablations_robustness.png',[
['信号扰动','AEIC 字段组合变化后，集体准确率保持在 87.3% 至 90.0%。'],['历史扰动','重置、隐藏或打乱历史后，结果保持在 84.7% 至 90.0%。'],['Panel 规模','3 至 5 个 Agent 时准确率为 87.3% 至 88.7%。稳定不等于每个字段独立不可缺少。']]);
q('主结果保持稳定，跨模型迁移存在边界').innerHTML=layout('研究内容及进展｜外部有效性边界','合同效果不能直接跨模型迁移','留出后端检验相同合同语义能否在不同模型家族中复现','assets/fig_s2_crossmodel.png',[
['行为差异','不同后端在首轮接受率、少数意见处理和方差收缩上表现不同。'],['确认性边界','留出 GLM 后端为 69.33% 对 70.67%，p=0.791，未复现主后端增益。'],['适用范围','公开合同需要针对后端重新校准。']]);
const inn=q('创新点与阶段性成果');
const paper=document.createElement('section');paper.className='s';paper.innerHTML=`<div class="head"><span>研究内容及进展｜阶段性成果</span><span></span></div><h2>论文、专利与实验资产</h2><p class="sub">阶段性成果均对应可复查的论文、代码、数据和系统记录</p><div class="paper-layout"><img src="assets/paper_first_page.png" alt="英文论文首页"><div><h3 style="font-size:1.4vw;color:#173f73">Making Effort Observable</h3><ul><li>英文论文正文已完成，正在推进投递。</li><li>中文论文理论部分已形成，与影视应用章节分开组织。</li><li>八组冻结实验包含 5,850 条问题级记录，保留版本、清单与独立复算结果。</li><li>专利文本已完成，后续按流程提交。</li></ul></div></div>`;inn.after(paper);
q('机制在影视搜索推荐中的落地').innerHTML=layout('系统实现及验证｜系统架构','影视搜索推荐系统架构','真实目录负责候选召回，多角色 Agent 负责偏好、覆盖与风险审议','assets/fig_system_architecture.png',[
['规划与检索','请求规划器分配风险、角色与预算；目录检索从约 18,430 部影片中生成候选。'],['共识机制','Preference、Diversity 和 Critic Agent 生成 AEIC 报告，控制器检查约束并结算信用。'],['审计输出','系统输出 Top-K、理由、异议、成本和完整 Trace；单一 DeepSeek API 承载多个逻辑角色。']]);
const ss=[...document.querySelectorAll('.s')];ss.forEach((s,i)=>{const x=s.querySelector('.head span:last-child');if(x)x.textContent=String(i+1).padStart(2,'0')+' / '+ss.length});
