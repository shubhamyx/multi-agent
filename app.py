from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

class AgentState(TypedDict):
    topic: str
    research: str
    article: str
    feedback: str
    final: str

def researcher(state: AgentState) -> AgentState:
    print("\n[RESEARCHER] Researching topic...")
    response = llm.invoke([
        SystemMessage(content="You are an expert researcher. Research the given topic thoroughly and provide key facts, statistics, and insights."),
        HumanMessage(content=f"Research this topic: {state['topic']}")
    ])
    state["research"] = response.content
    return state

def writer(state: AgentState) -> AgentState:
    print("\n[WRITER] Writing article...")
    response = llm.invoke([
        SystemMessage(content="You are an expert writer. Write a well-structured article based on the research provided."),
        HumanMessage(content=f"Write an article about '{state['topic']}' using this research:\n{state['research']}")
    ])
    state["article"] = response.content
    return state

def critic(state: AgentState) -> AgentState:
    print("\n[CRITIC] Reviewing article...")
    response = llm.invoke([
        SystemMessage(content="You are a harsh editor. Review the article and provide specific improvements needed."),
        HumanMessage(content=f"Review this article and suggest improvements:\n{state['article']}")
    ])
    state["feedback"] = response.content
    return state

def editor(state: AgentState) -> AgentState:
    print("\n[EDITOR] Finalizing article...")
    response = llm.invoke([
        SystemMessage(content="You are a senior editor. Improve the article based on the feedback provided."),
        HumanMessage(content=f"Improve this article:\n{state['article']}\n\nBased on this feedback:\n{state['feedback']}")
    ])
    state["final"] = response.content
    return state

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("critic", critic)
workflow.add_node("editor", editor)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")
workflow.add_edge("critic", "editor")
workflow.add_edge("editor", END)

app = workflow.compile()

def run_pipeline(topic):
    print(f"\nStarting multi-agent pipeline for: {topic}")
    print("=" * 50)
    
    result = app.invoke({
        "topic": topic,
        "research": "",
        "article": "",
        "feedback": "",
        "final": ""
    })
    
    print("\n" + "=" * 50)
    print("FINAL ARTICLE:")
    print("=" * 50)
    print(result["final"])

import gradio as gr

def gradio_pipeline(topic):
    if not topic.strip():
        return "", "", "", ""
    
    result = app.invoke({
        "topic": topic,
        "research": "",
        "article": "",
        "feedback": "",
        "final": ""
    })
    
    return result["research"], result["article"], result["feedback"], result["final"]

with gr.Blocks(title="Multi-Agent Pipeline") as ui:
    gr.Markdown("# Multi-Agent Research & Writing Pipeline")
    gr.Markdown("Enter a topic and watch 4 AI agents collaborate to produce a polished article.")
    
    topic_input = gr.Textbox(label="Topic", placeholder="e.g. AI in Healthcare")
    run_btn = gr.Button("Run Pipeline", variant="primary")
    
    with gr.Tabs():
        with gr.Tab("Research"):
            research_out = gr.Markdown()
        with gr.Tab("Draft Article"):
            article_out = gr.Markdown()
        with gr.Tab("Critic Feedback"):
            feedback_out = gr.Markdown()
        with gr.Tab("Final Article"):
            final_out = gr.Markdown()
    
    run_btn.click(
        fn=gradio_pipeline,
        inputs=[topic_input],
        outputs=[research_out, article_out, feedback_out, final_out]
    )

ui.launch()