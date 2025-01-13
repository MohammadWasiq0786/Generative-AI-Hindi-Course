import os
from dotenv import load_dotenv, find_dotenv
import openai
import asyncio
import streamlit as st
from typing import List, Dict
from praisonaiagents import Agent, Task, PraisonAIAgents, TaskOutput
from duckduckgo_search import DDGS
from pydantic import BaseModel

# Load environment variables from the .env file
_ = load_dotenv(find_dotenv())

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Set the API key for OpenAI
openai.api_key = openai_api_key





# 1. Define async tool
class SearchResult(BaseModel):
    query: str
    results: List[Dict[str, str]]
    total_results: int

async def async_search_tool(query: str) -> Dict:
    """Perform asynchronous search and return structured results."""
    await asyncio.sleep(1)  # Simulate network delay
    try:
        results = []
        ddgs = DDGS()
        for result in ddgs.text(keywords=query, max_results=5):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        print(f"Error during async search: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0
        }




# 2. Create agents
async_agent = Agent(
    name="AsyncSearchAgent",
    role="Search Specialist",
    goal="Perform fast parallel searches with structured results",
    backstory="Expert in efficient data retrieval and parallel search operations",
    tools=[async_search_tool],
    self_reflect=False,
    verbose=True,
    markdown=True
)

summary_agent = Agent(
    name="SummaryAgent",
    role="Research Synthesizer",
    goal="Create concise summaries from multiple search results",
    backstory="Expert in analyzing and synthesizing information from multiple sources",
    self_reflect=True,
    verbose=True,
    markdown=True
)



# 3. Create tasks

async_task = Task(
    name="async_search",
    description="Search for 'Async programming' and return results in JSON format with query, results array, and total_results count.",
    expected_output="SearchResult model with structured data",
    agent=async_agent,
    async_execution=True,
    output_json=SearchResult
)

async def run_parallel_tasks(): 
    """Run multiple async tasks in parallel"""
    print("\nRunning Parallel Async Tasks...")
    
    # Define different search topics
    search_topics = [
        "Latest AI Developments 2024",
        "Machine Learning Best Practices",
        "Neural Networks Architecture"
    ]
    
    # Create tasks for different topics
    parallel_tasks = [
        Task(
            name=f"search_task_{i}",
            description=f"Search for '{topic}' and return structured results with query details and findings.",
            expected_output="SearchResult model with search data",
            agent=async_agent,
            async_execution=True,
            output_json=SearchResult
        ) for i, topic in enumerate(search_topics)
    ]
    
    # Create summarization task
    summary_task = Task(
        name="summary_task",
        description="Analyze all search results and create a concise summary highlighting key findings, patterns, and implications.",
        expected_output="Structured summary with key findings and insights",
        agent=summary_agent,
        async_execution=False,
        context=parallel_tasks
    )
    
    # 4. Start Agents
    agents = PraisonAIAgents(
        agents=[async_agent, summary_agent],
        tasks=parallel_tasks + [summary_task],
        verbose=1,
        process="sequential"
    )
    
    # Run all tasks
    results = await agents.astart()
    
    # Return results in a serializable format
    return {
        "search_results": {
            "task_status": {k: v for k, v in results["task_status"].items() if k != summary_task.id},
            "task_results": [str(results["task_results"][i]) if results["task_results"][i] else None 
                           for i in range(len(parallel_tasks))]
        },
        "summary": str(results["task_results"][summary_task.id]) if results["task_results"].get(summary_task.id) else None,
        "topics": search_topics
    }
    
# 5. Run Async Function
async def main():
    """Main execution function"""
    print("Starting Async AI Agents Examples...")
    
    try:
        results = await run_parallel_tasks()
        # Display results in Streamlit
        st.title("Search Results")
        st.header("Search Topics")
        st.write(results["topics"])

        st.header("Search Results")
        for i, result in enumerate(results["search_results"]["task_results"]):
            st.subheader(f"Topic {i+1}")
            st.json(result)

        st.header("Summary")
        st.write(results["summary"] or "No summary generated.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        st.error(f"Error in main execution: {e}")

# 6. Run Streamlit
if __name__ == "__main__":
    # Run the main function with Streamlit
    asyncio.run(main())
