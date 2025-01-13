from praisonaiagents import Agent, Task, PraisonAIAgents
from praisonaiagents.tools import duckduckgo

research_agent = Agent(
    role="Research Analyst",
    goal="Research and document key information about topics",
    backstory="Expert at analyzing and storing information in memory",
    llm="gpt-4o-mini",
    tools=[duckduckgo]
)

blog_agent = Agent(
    role="Blog Writer",
    goal="Write a blog post about the research",
    backstory="Expert at writing blog posts",
    llm="gpt-4o-mini"
)

research_task = Task(
    description="Research and document key information about topics",
    agent=research_agent
)

blog_task = Task(
    description="Write a blog post about the research",
    agent=blog_agent
)

agents = PraisonAIAgents(
    agents=[research_agent, blog_agent],
    tasks=[research_task, blog_task],
    memory=True
)   

results = agents.start()

import streamlit as st

# Displaying Task Status
st.title("AI Agents Task Results")
st.header("Task Status")

for task_id, status in results["task_status"].items():
    st.write(f"**Task {task_id}:** {status}")

# Displaying Task Results
st.header("Task Results")
for task_id, result in results["task_results"].items():
    with st.expander(f"Task {task_id}: {result.description}"):
        st.subheader("Summary")
        st.write(result.summary)

        st.subheader("Output")
        st.markdown("### Key Findings and Details:")
        st.text_area("Detailed Output", value=result.raw, height=300, key=f"output_{task_id}")



