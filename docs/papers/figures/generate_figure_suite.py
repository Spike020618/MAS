"""Generate the four-figure AEIC mechanism architecture suite."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = Path(__file__).resolve().parent
NAVY, BLUE, TEAL, ORANGE = "#18324A", "#3F73A5", "#2C827A", "#CB6F2B"
LBLUE, LTEAL, LORANGE, LGRAY = "#EAF2F8", "#E7F3F1", "#F8EEE6", "#F3F5F7"
GRAY, LINE, WHITE = "#5F6B77", "#AAB5BF", "#FFFFFF"
PURPLE, LPURPLE, CORAL, LCORAL, GREEN, LGREEN, RED, LRED = (
    "#7557A5", "#F0EBF8", "#D95D55", "#FBECE9", "#5E963B", "#EDF6E8", "#BC4450", "#FAEAEC")
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "svg.fonttype":"none", "pdf.fonttype":42})

def rect(ax,x,y,w,h,title,body="",fc=WHITE,ec=LINE,tc=NAVY,fs=9,bfs=7.2,r=.012,lw=1.2):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=.006,rounding_size={r}",facecolor=fc,edgecolor=ec,linewidth=lw)
    ax.add_patch(p)
    ax.text(x+w/2,y+h-.018,title,ha="center",va="top",fontsize=fs,fontweight="bold",color=tc)
    if body: ax.text(x+w/2,y+h*.44,body,ha="center",va="center",fontsize=bfs,color=GRAY,linespacing=1.35)
    return p

def arr(ax,a,b,c=BLUE,lw=1.5,dash=False,rad=0,txt=None,tpos=.5):
    p=FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,color=c,linewidth=lw,
                      linestyle="--" if dash else "-",connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(p)
    if txt:
        x=a[0]+(b[0]-a[0])*tpos; y=a[1]+(b[1]-a[1])*tpos
        ax.text(x,y+.014,txt,ha="center",va="bottom",fontsize=6.8,color=c,
                bbox=dict(facecolor=WHITE,edgecolor="none",pad=.7))

def canvas(title,subtitle,size=(15.5,8.3)):
    fig,ax=plt.subplots(figsize=size); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(.03,.982,title,ha="left",va="top",fontsize=15,fontweight="bold",color=NAVY)
    ax.text(.03,.942,subtitle,ha="left",va="top",fontsize=9,color=GRAY)
    return fig,ax

def save(fig,name):
    for ext,kw in (("svg",{}),("pdf",{}),("png",{"dpi":240})):
        fig.savefig(OUT/f"{name}.{ext}",bbox_inches="tight",pad_inches=.08,**kw)
    plt.close(fig)

def person_icon(ax,x,y,color):
    ax.add_patch(Circle((x,y+.018),.011,fc=color,ec="none"))
    ax.add_patch(FancyBboxPatch((x-.017,y-.018),.034,.027,boxstyle="round,pad=.002,rounding_size=.006",fc=color,ec="none"))

def doc_icon(ax,x,y,color):
    ax.add_patch(Rectangle((x-.014,y-.018),.028,.037,fc=WHITE,ec=color,lw=1.2))
    for dy in (.009,0,-.009): ax.plot([x-.009,x+.008],[y+dy,y+dy],color=color,lw=1)

def branch_icon(ax,x,y,color):
    ax.add_patch(Circle((x,y+.016),.006,fc=color));
    for dx in (-.016,0,.016):
        ax.plot([x,x+dx],[y+.010,y-.010],color=color,lw=1.2)
        ax.add_patch(Circle((x+dx,y-.014),.005,fc=WHITE,ec=color,lw=1))

def rich_overview():
    fig,ax=canvas("AEIC mechanism: from public contract to auditable collective decision",
        "Observable actions connect individual follower choices, multi-round collaboration, and mechanism-level evaluation.",(15.8,8.6))
    # top input ribbon
    ax.add_patch(FancyBboxPatch((.03,.83),.94,.075,boxstyle="round,pad=.006,rounding_size=.012",fc=LBLUE,ec=BLUE,lw=1.2))
    inputs=[(.08,"TASK / QUESTION","x + answer space"),(.29,"PUBLIC CONTRACT","reward · cost · policy"),(.53,"PANEL CONFIG","agents · rounds · seed"),(.76,"INFORMATION RULE","who observes what, when")]
    for x,t,b in inputs:
        doc_icon(ax,x,.867,BLUE); ax.text(x+.025,.875,t,fontsize=7.6,fontweight="bold",color=NAVY,va="center"); ax.text(x+.025,.853,b,fontsize=6.5,color=GRAY,va="center")
    # follower cards
    ax.text(.035,.79,"A. FOLLOWER MICRO-ARCHITECTURE",fontsize=8.5,fontweight="bold",color=PURPLE)
    agent_specs=[(.035,.58,"AGENT 1",PURPLE,LPURPLE,"analyst"),(.035,.365,"AGENT 2",CORAL,LCORAL,"skeptic"),(.035,.15,"AGENT 3",GREEN,LGREEN,"verifier")]
    for x,y,name,c,fc,role in agent_specs:
        ax.add_patch(FancyBboxPatch((x,y),.24,.17,boxstyle="round,pad=.006,rounding_size=.012",fc=WHITE,ec=c,lw=1.4))
        ax.add_patch(FancyBboxPatch((x+.012,y+.093),.063,.06,boxstyle="round,pad=.004,rounding_size=.008",fc=fc,ec="none"))
        person_icon(ax,x+.043,y+.118,c); ax.text(x+.085,y+.139,name,fontsize=8.2,fontweight="bold",color=c); ax.text(x+.085,y+.116,role,fontsize=6.6,color=GRAY)
        chips=[("OBSERVE",LBLUE,BLUE),("CHOOSE",LPURPLE,PURPLE),("EXECUTE",LORANGE,ORANGE),("REPORT",LTEAL,TEAL)]
        for j,(t,bg,cc) in enumerate(chips):
            xx=x+.012+j*.056
            ax.add_patch(FancyBboxPatch((xx,y+.023),.05,.046,boxstyle="round,pad=.002,rounding_size=.005",fc=bg,ec=cc,lw=.8))
            ax.text(xx+.025,y+.046,t,ha="center",va="center",fontsize=5.6,fontweight="bold",color=cc)
        ax.text(x+.12,y+.082,"offer m  →  action aᵢ  →  AEIC report rᵢ",ha="center",fontsize=6.3,color=GRAY)
    # central mechanism
    ax.text(.315,.79,"B. LEADER–FOLLOWER CONTROL PLANE",fontsize=8.5,fontweight="bold",color=TEAL)
    ax.add_patch(FancyBboxPatch((.31,.15),.35,.60,boxstyle="round,pad=.009,rounding_size=.018",fc="#F7FBFA",ec=TEAL,lw=1.5))
    # commitment + router
    rect(ax,.345,.655,.28,.065,"① Leader commitment","render visible credit–cost offer before generation",LBLUE,BLUE,BLUE,8.5,6.6)
    rect(ax,.345,.555,.28,.065,"② Action router","declared action selects an enforceable resource path",LPURPLE,PURPLE,PURPLE,8.5,6.6)
    arr(ax,(.485,.655),(.485,.62),TEAL,1.2)
    # action paths
    paths=[(.33,"LOW","short",BLUE,LBLUE),(.407,"AEIC","full",PURPLE,LPURPLE),(.484,"VERIFY","audit",ORANGE,LORANGE),(.561,"IMITATE","peer",CORAL,LCORAL)]
    for x,t,b,c,fc in paths:
        ax.add_patch(FancyBboxPatch((x,.465),.067,.06,boxstyle="round,pad=.003,rounding_size=.007",fc=fc,ec=c,lw=1))
        ax.text(x+.0335,.503,t,ha="center",fontsize=6.4,fontweight="bold",color=c); ax.text(x+.0335,.479,b,ha="center",fontsize=5.8,color=GRAY)
        arr(ax,(.485,.555),(x+.0335,.525),c,.9)
    # AEIC report grid
    ax.text(.485,.43,"③ STRUCTURED REPORT INTERFACE",ha="center",fontsize=7.2,fontweight="bold",color=TEAL)
    aeic=[("A","Assumption",BLUE,LBLUE),("E","Evidence",GREEN,LGREEN),("I","Inference",PURPLE,LPURPLE),("C","Conclusion",ORANGE,LORANGE)]
    for j,(letter,name,c,fc) in enumerate(aeic):
        x=.345+j*.07
        ax.add_patch(FancyBboxPatch((x,.355),.062,.055,boxstyle="round,pad=.002,rounding_size=.006",fc=fc,ec=c,lw=1))
        ax.text(x+.012,.383,letter,ha="center",va="center",fontsize=9,fontweight="bold",color=c); ax.text(x+.039,.383,name,ha="center",va="center",fontsize=5.5,color=GRAY)
    # settlement and state
    rect(ax,.345,.255,.13,.065,"④ Settlement","quality credit − action cost",LGREEN,GREEN,GREEN,8,6.2)
    rect(ax,.495,.255,.13,.065,"⑤ Persistent state","ρᵢ · trace · tokens · latency",LGRAY,NAVY,NAVY,8,6.2)
    arr(ax,(.485,.355),(.41,.32),GREEN,1.1); arr(ax,(.475,.287),(.495,.287),NAVY,1.1)
    rect(ax,.345,.165,.28,.060,"Round controller","synchronous barrier · rollback · advance",WHITE,TEAL,TEAL,7.6,5.8)
    arr(ax,(.56,.255),(.56,.22),TEAL,1.0); arr(ax,(.345,.195),(.29,.235),LINE,1.0,True,.2,"hᵢᵗ⁺¹")
    # report/action arrows from agents to plane and offers back
    for _,y,_,c,_,_ in agent_specs:
        arr(ax,(.345,.685),(.275,y+.13),BLUE,1.0,False,.12)
        arr(ax,(.275,y+.07),(.345,.385),c,1.0,False,-.10)
    # right collaboration
    ax.text(.695,.79,"C. MULTI-AGENT COLLABORATION",fontsize=8.5,fontweight="bold",color=ORANGE)
    ax.add_patch(FancyBboxPatch((.69,.15),.28,.60,boxstyle="round,pad=.009,rounding_size=.018",fc="#FFFBF8",ec=ORANGE,lw=1.5))
    # information chips
    ax.text(.83,.708,"INFORMATION ENVIRONMENTS",ha="center",fontsize=7.5,fontweight="bold",color=NAVY)
    envs=[("Independent",BLUE,LBLUE),("Evidence",GREEN,LGREEN),("Social",PURPLE,LPURPLE),("Copy",CORAL,LCORAL),("Counter-E",ORANGE,LORANGE)]
    for j,(t,c,fc) in enumerate(envs):
        x=.71+(j%3)*.083; y=.65-(j//3)*.055
        ax.add_patch(FancyBboxPatch((x,y),.074,.041,boxstyle="round,pad=.002,rounding_size=.006",fc=fc,ec=c,lw=.8)); ax.text(x+.037,y+.0205,t,ha="center",va="center",fontsize=5.7,fontweight="bold",color=c)
    # round timeline
    ax.text(.83,.555,"SYNCHRONOUS DELIBERATION",ha="center",fontsize=7.5,fontweight="bold",color=NAVY)
    for j,(t,c) in enumerate((("R1",BLUE),("R2",PURPLE),("R3",ORANGE))):
        x=.745+j*.085; ax.add_patch(Circle((x,.505),.029,fc=WHITE,ec=c,lw=1.5)); ax.text(x,.505,t,ha="center",va="center",fontsize=7,fontweight="bold",color=c)
        if j<2: arr(ax,(x+.029,.505),(x+.056,.505),NAVY,1.0)
    ax.text(.83,.465,"t reads permitted reports from t−1; same-round isolation",ha="center",fontsize=6.2,color=GRAY)
    # decision + audit cards
    rect(ax,.72,.36,.22,.065,"Majority aggregation","explicit C fields → final decision",LORANGE,ORANGE,ORANGE,8.2,6.4)
    audits=[(.71,"Behavior","effort · flips",PURPLE,LPURPLE),(.795,"Decision","accuracy · pairs",TEAL,LTEAL),(.88,"Cost / audit","tokens · failures",NAVY,LGRAY)]
    for x,t,b,c,fc in audits: rect(ax,x,.225,.075,.09,t,b,fc,c,c,6.8,5.7)
    arr(ax,(.83,.465),(.83,.425),ORANGE,1.1)
    for x,_,_,c,_ in audits: arr(ax,(.83,.36),(x+.037,.315),c,.9)
    # risk + fallback strip
    ax.add_patch(FancyBboxPatch((.71,.17),.23,.032,boxstyle="round,pad=.002,rounding_size=.005",fc=LRED,ec=RED,lw=.8))
    ax.text(.825,.186,"failure path: parse / API error → retry / rollback / fallback",ha="center",va="center",fontsize=5.8,color=RED,fontweight="bold")
    arr(ax,(.66,.385),(.69,.385),NAVY,1.3,False,0,"reports + policy")
    # bottom claim strip
    ax.add_patch(FancyBboxPatch((.03,.055),.94,.055,boxstyle="round,pad=.005,rounding_size=.009",fc=WHITE,ec=LINE,lw=1))
    items=[("Follower response","high-effort action"),("Decision quality","paired accuracy"),("Resource cost","tokens / calls"),("Validity boundary","held-out backend")]
    for j,(a,b) in enumerate(items):
        x=.055+j*.235; ax.text(x,.085,a,fontsize=6.8,fontweight="bold",color=(BLUE,TEAL,ORANGE,RED)[j]); ax.text(x,.067,b,fontsize=6.2,color=GRAY)
    save(fig,"fig1_aeic_md_overview_rich")

def agent_card(ax,y,idx):
    x,w,h=.025,.225,.205
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.005,rounding_size=.012",fc=WHITE,ec=BLUE,lw=1.3))
    ax.text(x+.012,y+h-.018,f"FOLLOWER AGENT {idx}",fontsize=8.5,fontweight="bold",color=BLUE,va="top")
    # internal causal stack
    labels=[("Observe", "task x · offer m · history hᵢᵗ", LBLUE),
            ("Choose action", "LOW / AEIC / VERIFY / IMITATE", LTEAL),
            ("Execute", "action-specific prompt + resource", LORANGE),
            ("Report", "AEIC = {A, E, I, C}", LGRAY)]
    yy=y+h-.052
    for k,(t,b,fc) in enumerate(labels):
        rh=.032
        ax.add_patch(Rectangle((x+.012,yy-rh),w-.024,rh,fc=fc,ec="none"))
        ax.text(x+.02,yy-rh/2,t,ha="left",va="center",fontsize=6.8,fontweight="bold",color=NAVY)
        ax.text(x+w-.018,yy-rh/2,b,ha="right",va="center",fontsize=6.1,color=GRAY)
        yy-=rh+.006
    return x+w,y+h/2

def overview():
    fig,ax=canvas("AEIC mechanism: mechanism-mediated multi-agent deliberation",
        "A public leader rule changes what followers choose to do before reports are aggregated.")
    # left: repeated agents
    ax.text(.025,.885,"FOLLOWER MICRO-ARCHITECTURE",fontsize=8.5,fontweight="bold",color=BLUE)
    ports=[]
    for i,y in enumerate((.64,.405,.17),1): ports.append(agent_card(ax,y,i))
    # center mechanism
    ax.add_patch(FancyBboxPatch((.29,.16),.31,.69,boxstyle="round,pad=.01,rounding_size=.02",fc=LTEAL,ec=TEAL,lw=1.5))
    ax.text(.445,.825,"LEADER–FOLLOWER MECHANISM",ha="center",fontsize=11,fontweight="bold",color=TEAL)
    rect(ax,.325,.70,.24,.09,"1  Public contract","credit rule gₘ · costs cₘ(a)\ninformation policy · verification budget",LBLUE,BLUE,BLUE,8.5,6.8)
    rect(ax,.325,.57,.24,.09,"2  Action enforcement","declared action selects the actual\nprompt, token limit, and resource path",LORANGE,ORANGE,ORANGE,8.5,6.8)
    rect(ax,.325,.44,.24,.09,"3  Observable settlement","AEIC completeness · valid conclusion\ncredit − cost → realized utility",WHITE,TEAL,TEAL,8.5,6.8)
    rect(ax,.325,.31,.24,.09,"4  Persistent state","offer · action trace · raw report\nreputation ρᵢ · tokens · latency",LGRAY,LINE,NAVY,8.5,6.8)
    rect(ax,.325,.19,.24,.075,"Round controller","synchronous rounds · rollback on failure",WHITE,TEAL,TEAL,8.5,6.7)
    for y1,y2 in ((.70,.66),(.57,.53),(.44,.40),(.31,.265)): arr(ax,(.445,y1),(.445,y2),TEAL,1.2)
    # broadcast and reports
    for px,py in ports:
        arr(ax,(.325,.745),(px,py+.045),BLUE,1.1,False,-.05,"offer m",.62)
        arr(ax,(px,py-.04),(.325,.475),TEAL,1.1,False,.06,"aᵢ, rᵢ",.50)
    arr(ax,(.325,.225),(.25,.23),LINE,1.1,True,.12,"hᵢᵗ⁺¹",.5)
    # right deliberation and outcomes
    ax.add_patch(FancyBboxPatch((.64,.16),.335,.69,boxstyle="round,pad=.01,rounding_size=.02",fc=LGRAY,ec=NAVY,lw=1.4))
    ax.text(.807,.825,"COLLECTIVE DELIBERATION + AUDIT",ha="center",fontsize=11,fontweight="bold",color=NAVY)
    rect(ax,.675,.70,.265,.09,"Information environments","independent · evidence update · social only\ncopy pressure · counterevidence",WHITE,BLUE,BLUE,8.5,6.8)
    rect(ax,.675,.56,.265,.10,"Synchronous interaction","round t reads only permitted reports from t−1\nsame-round agents remain mutually hidden",WHITE,TEAL,TEAL,8.5,6.8)
    rect(ax,.675,.43,.265,.08,"Collective decision","explicit C fields → round-3 majority vote",WHITE,ORANGE,ORANGE,8.5,6.8)
    rect(ax,.665,.22,.085,.145,"Behavior","effort rate\nactions\nstance flips\nnovelty",LBLUE,BLUE,BLUE,8,6.5)
    rect(ax,.765,.22,.085,.145,"Decision","accuracy\ncorrected\nharmed\ndomain effect",LTEAL,TEAL,TEAL,8,6.5)
    rect(ax,.865,.22,.085,.145,"Cost / audit","tokens\nlatency\nfailures\nhashes",LORANGE,ORANGE,ORANGE,8,6.5)
    arr(ax,(.60,.615),(.675,.615),NAVY,1.5,False,0,"reports + policy")
    arr(ax,(.807,.70),(.807,.66),NAVY,1.2); arr(ax,(.807,.56),(.807,.51),NAVY,1.2)
    for x in (.707,.807,.907): arr(ax,(.807,.43),(x,.365),NAVY,1.0)
    # identifying strip
    ax.add_patch(FancyBboxPatch((.29,.055),.685,.065,boxstyle="round,pad=.006,rounding_size=.01",fc=WHITE,ec=LINE,lw=1))
    ax.text(.31,.098,"IDENTIFICATION",fontsize=7.5,fontweight="bold",color=NAVY)
    ax.text(.31,.072,"same question · seed · panel · rounds · information; only the public mechanism changes",
            fontsize=7.2,color=GRAY)
    save(fig,"fig1_aeic_md_overview")

def runtime_sequence():
    fig,ax=canvas("AEIC mechanism runtime: one question across three synchronous rounds",
        "The action is chosen before solving; settlement and history are produced after execution.",(15.5,7.8))
    lanes=[("Leader / controller",.76,BLUE,LBLUE),("Follower agents",.55,TEAL,LTEAL),("Executor",.34,ORANGE,LORANGE),("Audit + aggregation",.13,NAVY,LGRAY)]
    for label,y,c,fc in lanes:
        ax.add_patch(Rectangle((.03,y),.94,.16,fc=fc,ec=c,lw=1))
        ax.text(.045,y+.13,label,fontsize=8,fontweight="bold",color=c,va="top")
    # role glyphs make the lanes read as interacting entities rather than rows in a table
    doc_icon(ax,.055,.805,BLUE); person_icon(ax,.055,.595,TEAL)
    branch_icon(ax,.055,.385,ORANGE); doc_icon(ax,.055,.175,NAVY)
    xs=[.16,.35,.54,.73]
    heads=["Commit","Choose","Execute","Settle"]
    for x,h in zip(xs,heads):
        ax.text(x,.755,h,ha="center",fontsize=8,fontweight="bold",color=NAVY)
        ax.plot([x,x],[.10,.93],color=LINE,lw=.7,ls=":")
    # round 1
    rect(ax,.09,.795,.14,.075,"Public offer m","rule · costs · policy",WHITE,BLUE,BLUE,8,6.5)
    rect(ax,.28,.585,.14,.075,"Declare aᵢ¹","before solving",WHITE,TEAL,TEAL,8,6.5)
    rect(ax,.47,.375,.14,.075,"Run selected path","produce rᵢ¹={A,E,I,C}",WHITE,ORANGE,ORANGE,8,6.5)
    rect(ax,.66,.165,.14,.075,"Settle + log","uᵢ¹ · ρᵢ¹ · trace",WHITE,NAVY,NAVY,8,6.5)
    for a,b,c in [((.23,.832),(.28,.622),BLUE),((.42,.622),(.47,.412),TEAL),((.61,.412),(.66,.202),ORANGE)]: arr(ax,a,b,c,1.3)
    # repeat block rounds 2/3
    ax.add_patch(FancyBboxPatch((.81,.55),.15,.32,boxstyle="round,pad=.006,rounding_size=.012",fc=WHITE,ec=TEAL,lw=1.2))
    ax.text(.885,.84,"ROUNDS 2–3",ha="center",fontsize=8.5,fontweight="bold",color=TEAL)
    ax.text(.885,.77,"read permitted r⁽ᵗ⁻¹⁾\nobserve updated ρᵢ\nchoose aᵢᵗ\nexecute and settle",
            ha="center",va="center",fontsize=7,color=GRAY,linespacing=1.5)
    arr(ax,(.73,.202),(.81,.60),NAVY,1.2,False,.15,"history",.55)
    rect(ax,.81,.165,.15,.075,"Final decision","majority over Cᵢ³",WHITE,NAVY,NAVY,8,6.5)
    arr(ax,(.885,.55),(.885,.24),NAVY,1.3)
    # visible protocol messages and failure side path
    for x,y,t,c,fc in ((.235,.73,"OFFER m",BLUE,LBLUE),(.425,.51,"ACTION aᵢ",TEAL,LTEAL),(.615,.29,"REPORT rᵢ",ORANGE,LORANGE)):
        ax.add_patch(FancyBboxPatch((x,y),.072,.027,boxstyle="round,pad=.002,rounding_size=.012",fc=fc,ec=c,lw=.8))
        ax.text(x+.036,y+.0135,t,ha="center",va="center",fontsize=5.7,fontweight="bold",color=c)
    ax.add_patch(FancyBboxPatch((.63,.31),.155,.035,boxstyle="round,pad=.003,rounding_size=.006",fc=LRED,ec=RED,lw=.8))
    ax.text(.707,.327,"parse/API error → retry or rollback",ha="center",va="center",fontsize=5.8,fontweight="bold",color=RED)
    arr(ax,(.707,.345),(.54,.385),RED,.9,True,-.08)
    # invariants
    ax.text(.03,.055,"CAUSAL INVARIANTS",fontsize=7.5,fontweight="bold",color=NAVY)
    ax.text(.145,.055,"offer precedes action",fontsize=7,color=GRAY)
    ax.text(.34,.055,"same-round isolation",fontsize=7,color=GRAY)
    ax.text(.535,.055,"ground truth withheld until final settlement",fontsize=7,color=GRAY)
    ax.text(.79,.055,"retries remain auditable",fontsize=7,color=GRAY)
    save(fig,"fig2_runtime_sequence")

def experimental_design():
    fig,ax=canvas("Experimental identification and evidence hierarchy",
        "Confirmatory effects are separated from mechanism checks, robustness, and external-validity boundaries.",(15.5,7.8))
    # paired backbone
    rect(ax,.035,.64,.20,.20,"Paired task unit","same question + seed\nsame 5-agent panel\nsame 3 rounds\nsame information policy",LGRAY,LINE,NAVY,9.5,7.4)
    ax.text(.285,.865,"ONLY THE PUBLIC OFFER CHANGES",fontsize=8.5,fontweight="bold",color=NAVY)
    conditions=[(.285,"No Mechanism","no explicit credit",LBLUE,BLUE),(.505,"Outcome Only","reward final correctness",LORANGE,ORANGE),(.725,"Static AEIC","reward observable process",LTEAL,TEAL)]
    for x,t,b,fc,c in conditions:
        rect(ax,x,.68,.18,.12,t,b,fc,c,c,9,7)
    # condition badges emphasize the experimental contrast
    for x,n,c in ((.30,"C0",BLUE),(.52,"C1",ORANGE),(.74,"C2",TEAL)):
        ax.add_patch(Circle((x,.785),.014,fc=c,ec="none")); ax.text(x,.785,n,ha="center",va="center",fontsize=5.5,color=WHITE,fontweight="bold")
    # one paired task fans out to the three alternative public offers
    ax.plot([.235,.265],[.74,.835],color=NAVY,lw=1.2)
    ax.plot([.265,.815],[.835,.835],color=NAVY,lw=1.2)
    for x in (.375,.595,.815): arr(ax,(x,.835),(x,.80),NAVY,1.1)
    # outcomes
    ax.text(.285,.61,"PRIMARY PAIRED OUTCOMES",fontsize=8,fontweight="bold",color=NAVY)
    ax.plot([.375,.815],[.635,.635],color=NAVY,lw=1.1)
    for x in (.375,.595,.815): arr(ax,(x,.68),(x,.635),NAVY,1.0)
    rect(ax,.285,.49,.18,.09,"Follower response","high-effort choice rate",WHITE,BLUE,BLUE,8.5,6.8)
    rect(ax,.505,.49,.18,.09,"Collective decision","question-level accuracy",WHITE,TEAL,TEAL,8.5,6.8)
    rect(ax,.725,.49,.18,.09,"Resource consequence","tokens / agent-round",WHITE,ORANGE,ORANGE,8.5,6.8)
    for x in (.375,.595,.815): arr(ax,(.595,.635),(x,.58),NAVY,1.0)
    # headline estimates belong in the evidence picture, not only in prose
    ax.text(.375,.475,"0.46% → 56.93%",ha="center",fontsize=6.4,fontweight="bold",color=BLUE)
    ax.text(.595,.475,"+14.22 pp",ha="center",fontsize=6.4,fontweight="bold",color=TEAL)
    ax.text(.815,.475,"calls · tokens · latency",ha="center",fontsize=6.1,fontweight="bold",color=ORANGE)
    # evidence pyramid / suites
    ax.text(.035,.42,"EVIDENCE HIERARCHY — 8 FROZEN SUITES / 5,850 QUESTION-LEVEL RECORDS",fontsize=8.5,fontweight="bold",color=NAVY)
    blocks=[
      (.035,.25,.22,.12,"Confirmatory effect","formal_v10 · 450 paired questions\nMcNemar + Holm correction",NAVY,LGRAY),
      (.275,.25,.22,.12,"Mechanism checks","reward/cost dose response\n4 information environments",BLUE,LBLUE),
      (.515,.25,.22,.12,"Robustness + ablation","2 extra seeds · 3/4/5 agents\nsignal masks · history policies",TEAL,LTEAL),
      (.755,.25,.22,.12,"External boundary","held-out GLM backend\nmain effect does not transfer",ORANGE,LORANGE),
    ]
    for x,y,w,h,t,b,c,fc in blocks: rect(ax,x,y,w,h,t,b,fc,c,c,8.5,6.8)
    # evidence maturity rail: claim strength decreases toward external transfer
    ax.plot([.09,.91],[.215,.215],color=LINE,lw=2)
    for x,c in ((.145,NAVY),(.385,BLUE),(.625,TEAL),(.865,ORANGE)):
        ax.add_patch(Circle((x,.215),.010,fc=c,ec=WHITE,lw=1))
    ax.text(.04,.205,"SUPPORTED",fontsize=5.8,fontweight="bold",color=NAVY,va="top")
    ax.text(.93,.205,"BOUNDARY",fontsize=5.8,fontweight="bold",color=ORANGE,va="top",ha="right")
    # audit base
    ax.add_patch(FancyBboxPatch((.035,.09),.94,.09,boxstyle="round,pad=.006,rounding_size=.012",fc=WHITE,ec=LINE,lw=1.1))
    audit=["matrix completeness","versioned schema","canonical JSONL","SHA-256 provenance","independent recomputation","0 summary mismatches"]
    for i,t in enumerate(audit):
        x=.06+i*.15
        ax.add_patch(Circle((x,.135),.012,fc=NAVY,ec="none")); ax.text(x,.135,"✓",ha="center",va="center",fontsize=7,color=WHITE,fontweight="bold")
        ax.text(x+.018,.135,t,ha="left",va="center",fontsize=6.8,color=GRAY)
    save(fig,"fig3_experimental_design")

if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    overview(); rich_overview(); runtime_sequence(); experimental_design()
    print("Generated AEIC mechanism figure suite")
