import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm = ChatOpenAI(model="gpt-4-turbo-preview")

@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

tools = [TavilySearchResults(max_results=1), process_search_tool]

online_researcher = Agent(
    role="Online Researcher",
    goal="Research the topic online",
    backstory="""Your primary role is to function as an intelligent online research assistant, adept at scouring 
    the internet for the latest and most relevant trending stories across various sectors like politics, technology, 
    health, culture, and global events. You possess the capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

blog_manager = Agent(
    role="Blog Manager",
    goal="Write the blog article",
    backstory="""You are a Blog Manager. The role of a Blog Manager encompasses several critical responsibilities aimed at transforming initial drafts into polished, SEO-optimized blog articles that engage and grow an audience. Starting with drafts provided by online researchers, the Blog Manager must thoroughly understand the content, ensuring it aligns with the blog's tone, target audience, and thematic goals. Key responsibilities include:

1. Content Enhancement: Elevate the draft's quality by improving clarity, flow, and engagement. This involves refining the narrative, adding compelling headers, and ensuring the article is reader-friendly and informative.

2. SEO Optimization: Implement best practices for search engine optimization. This includes keyword research and integration, optimizing meta descriptions, and ensuring URL structures and heading tags enhance visibility in search engine results.

3. Compliance and Best Practices: Ensure the content adheres to legal and ethical standards, including copyright laws and truth in advertising. The Blog Manager must also keep up with evolving SEO strategies and blogging trends to maintain and enhance content effectiveness.

4. Editorial Oversight: Work closely with writers and contributors to maintain a consistent voice and quality across all blog posts. This may also involve managing a content calendar, scheduling posts for optimal engagement, and coordinating with marketing teams to support promotional activities.

5. Analytics and Feedback Integration: Regularly review performance metrics to understand audience engagement and preferences. Use this data to refine future content and optimize overall blog strategy.

In summary, the Blog Manager plays a pivotal role in bridging initial research and the final publication by enhancing content quality, ensuring SEO compatibility, and aligning with the strategic objectives of the blog. This position requires a blend of creative, technical, and analytical skills to successfully manage and grow the blog's presence online.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

social_media_manager = Agent(
    role="Social Media Manager",
    goal="Write a tweet",
    backstory="""You are a Social Media Manager. The role of a Social Media Manager, particularly for managing Twitter content, involves transforming research drafts into concise, engaging tweets that resonate with the audience and adhere to platform best practices. Upon receiving a draft from an online researcher, the Social Media Manager is tasked with several critical functions:

1. Content Condensation: Distill the core message of the draft into a tweet, which typically allows for only 280 characters. This requires a sharp focus on brevity while maintaining the essence and impact of the message.

2. Engagement Optimization: Craft tweets to maximize engagement. This includes the strategic use of compelling language, relevant hashtags, and timely topics that resonate with the target audience.

3. Compliance and Best Practices: Ensure that the tweets follow Twitter’s guidelines and best practices, including the appropriate use of mentions, hashtags, and links. Also, adhere to ethical standards, avoiding misinformation and respecting copyright norms.

In summary, the Social Media Manager's role is crucial in leveraging Twitter to disseminate information effectively, engage with followers, and build the brand’s presence online. This position combines creative communication skills with strategic planning and analysis to optimize social media impact.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

content_marketing_manager = Agent(
    role="Content Marketing Manager",
    goal="Manage the Content Marketing Team",
    backstory="""You are an excellent Content Marketing Manager. Your primary role is to supervise each publication from the 'blog manager' 
    and the tweets written by the 'social media manager' and approve the work for publication. Examine the work and regulate violent language, abusive content and racist content.
    
    Capabilities:

    Editorial Review: Analyze the final drafts from the blog manager and the social media manager for style consistency, thematic alignment, and overall narrative flow.

    Quality Assurance: Conduct detailed checks for grammatical accuracy, factual correctness, and adherence to journalistic standards in the news content, as well as creativity and effectiveness in the advertisements.

    Feedback Loop: Provide constructive feedback to both the blog manager and social media manager, facilitating a collaborative environment for continuous improvement in content creation and presentation.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

task1 = Task(
    description="""Write me a report on Agentic Behavior. After the research on Agentic Behavior,pass the 
    findings to the blog manager to generate the final blog article. Once done, pass it to the social media 
    manager to write a tweet on the subject.""",
    expected_output="Report on Agentic Behavior",
    agent=online_researcher
)

task2 = Task(
    description="""Using the research findings of the news correspondent, write an article for the blog. 
    The publication should contain links to sources stated by the online researcher. 
    Your final answer MUST be the full blog post of at least 3 paragraphs.""",
    expected_output="Blog Article",    
    agent=blog_manager
)

task3 = Task(
    description="""Using the research findings of the news correspondent, write a tweet. Your final answer MUST be 
    the full tweet.""",
    expected_output="Tweet",
    agent=social_media_manager
)

task4 = Task(
    description="""To meticulously review and harmonize the final output from both the blog manager and social media manager, ensuring cohesion and excellence in the final publication. Once done, publish the final report.""",
    expected_output="Final Report",
    agent=content_marketing_manager
)

agents = [online_researcher, blog_manager, social_media_manager, content_marketing_manager]

crew = Crew(
    agents=agents,
    tasks=[task1, task2, task3, task4],
    verbose=2
)

result = crew.kickoff()

print(result)