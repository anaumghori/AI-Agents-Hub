# Import Libraries and Set env
import os
from helper import load_env
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
load_env()
os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
groq_llm = "groq/llama-3.1-70b-versatile"
search_tool = SerperDevTool()


# Creating Agents
market_researcher = Agent(
    role="Market Researcher",
    goal=(
        "Gather comprehensive stock market data for {companies} in {industries} between {start_date} and {end_date}. "
        "Focus on gathering data, including stock prices, recent news, and any notable trends during this period."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a seasoned market researcher with an eye for detail, capable of finding "
        "the latest trends and consolidating them into meaningful insights."
    ),
    tools=[search_tool],
    llm=os.environ['OPENAI_MODEL_NAME'],
)

financial_strategist = Agent(
    role="Financial Strategist",
    goal=(
        "Analyze the gathered data between {start_date} and {end_date}, identify risks, opportunities, "
        "and actionable strategies for {companies} in {industries}. Also recommend specific investment strategies, "
        "including risk levels (e.g., low, medium, high), diversification options, and long-term or short-term focus."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "As a financial strategist, you excel at turning data into actionable insights, helping clients "
        "navigate the complexities of the market with precision. You also specialize in crafting effective "
        "investment strategies tailored to specific industries."
    ),
    llm=groq_llm,
)

report_writer = Agent(
    role="Report Writer",
    goal=(
        "Create a professional, well-structured financial report based on the analysis of data gathered between {start_date} and {end_date}. "
        "The report should summarize the market research, analysis, and investment strategies, including an executive summary, detailed analysis, "
        "investment recommendations, and a conclusion with actionable insights."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a skilled financial writer, adept at presenting complex analyses in a clear and engaging format."
    ),
    llm=os.environ['OPENAI_MODEL_NAME'],
)


# Creating Tasks
market_research_task = Task(
    description=(
        "Research the latest stock market trends for {companies} in {industries} between {start_date} and {end_date}. "
        "Focus on gathering comprehensive data, including stock prices, recent news, and any notable trends during this period."
    ),
    expected_output="A detailed summary of the latest stock market data and trends.",
    tools=[search_tool],
    agent=market_researcher,
)

financial_analysis_task = Task(
    description=(
        "Analyze the market data and trends provided by the Market Researcher for {companies} in {industries} between {start_date} and {end_date}. "
        "Identify risks, opportunities, and actionable strategies. Additionally, recommend specific investment strategies, "
        "including risk levels (e.g., low, medium, high), diversification options, and long-term or short-term focus."
    ),
    expected_output="A structured analysis with key risks, opportunities, actionable insights, and specific investment strategies.",
    agent=financial_strategist,
)

report_writing_task = Task(
    description=(
        "Create a professional financial report summarizing the market research, analysis, and investment strategies for {companies} in {industries} "
        "between {start_date} and {end_date}. The report should include an executive summary, detailed analysis, investment recommendations, "
        "and a conclusion with actionable insights."
    ),
    expected_output="A professional financial report with actionable insights.",
    agent=report_writer,
)


# Creating and Running Crew
financial_analyst_crew = Crew(
    agents=[market_researcher, financial_strategist, report_writer],
    tasks=[market_research_task, financial_analysis_task, report_writing_task],
    process=Process.sequential,
)

inputs = {
    # List of companies for analysis
    "companies": "Apple, Tesla",

    # Industries to focus on
    "industries": "Technology, Automotive",

    # Start date of the analysis period
    "start_date": "1 October 2024",

    # End date of the analysis period
    "end_date": "20 December 2024",
}

results = financial_analyst_crew.kickoff(inputs=inputs)


# Final Result
from IPython.display import display, Markdown
display(Markdown(results.raw))

# Save the report
with open("financial_report.md", "w") as file:
    file.write(results.raw)
print("Financial report saved as 'Smart_Invest_report.md'.")
