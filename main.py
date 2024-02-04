from crewai import Agent, Task, Crew
import os
import openai

os.environ["OPENAI_API_KEY"] = "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"


def researcher_agent(topic):
    response = openai.Completion.create(
        engine="openhermes",
        prompt=f"Teach someone new about {topic}.",
        temperature=0.7,
        max_tokens=200,
        n=1,
    )
    return response.choices[0].message.content.strip()


def writer_agent(researcher_ideas):
    response = openai.Completion.create(
        engine="openhermes",
        prompt=f"{researcher_ideas}\nNow, explain the topic in detail.",
        temperature=0.7,
        max_tokens=500,
        n=1,
    )
    return response.choices[0].message.content.strip()


def examiner_agent(written_content):
    response = openai.Completion.create(
        engine="text-ollama-002",
        prompt=f"{written_content}\nNow, generate 2-3 test questions.",
        temperature=0.7,
        max_tokens=300,
        n=3,
    )
    questions = [choice.message.content.strip() for choice in response.choices]
    return questions


researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting
    actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[researcher_agent]
)


writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for
    your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    tools=[writer_agent]
)


examiner = Agent(
    role='Content Examiner',
    goal='Evaluate the quality and accuracy of the content',
    backstory="""You are an experienced Content Examiner, responsible for
    ensuring the accuracy and quality of tech-related content.
    Your keen eye for detail helps maintain the highest standards.""",
    verbose=True,
    allow_delegation=False,
    tools=[examiner_agent]
)


task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Your final answer MUST be a full analysis report""",
    agent=researcher
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
    post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.
    Your final answer MUST be the full blog post of at least 4 paragraphs.""",
    agent=writer
)

task3 = Task(
    description="""Review the blog post created by the writer.
    Evaluate its accuracy, quality, and adherence to the provided insights.
    Provide constructive feedback if needed.""",
    agent=examiner
)


crew = Crew(
    agents=[researcher, writer, examiner],
    tasks=[task1, task2, task3],
    verbose=2,
)


result = crew.kickoff()

print("######################")
print(result)
