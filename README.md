# Multi-Agent Research & Writing Pipeline

A multi-agent AI system where specialized agents collaborate to research and write articles.

## How it works
1. **Researcher** — researches the topic and gathers key facts
2. **Writer** — writes a structured article from the research
3. **Critic** — reviews the article and identifies weaknesses
4. **Editor** — improves the article based on critic feedback

## Tech Stack
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Agents**: LangGraph StateGraph
- **Framework**: LangChain

## Setup

1. Clone the repo
2. Create virtual environment

    python -m venv venv
    venv\Scripts\activate

3. Install dependencies

    pip install langchain langchain-groq langgraph python-dotenv

4. Create `.env` file

    GROQ_API_KEY=your_key_here

5. Run

    python app.py

## Built by
Shubham Yadav — building AI projects in public