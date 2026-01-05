"""
DSPy Signatures for View Selector and SQL Writer agents.
"""

import dspy
from typing import List


class ViewSelectorSignature(dspy.Signature):
    """
    You are an expert database view selector. Given a user query, you need to identify all database views that contain the relevant data and needed to answer the question.

    Key Instructions:
    1. Identify Relevant Views: Carefully analyze the QUESTION and the provided VIEW(s) descriptions, selectors and columns using your financial knowledge. 
       Determine which view(s) are most likely to contain the required data to answer the QUESTION.
    2. No Match Scenario: If none of the provided views appear relevant to the QUESTION, explicitly return ['<NO_VIEWS>'] instead of suggesting irrelevant views. 
       Remember, it's totally fine to say that it's not possible.
    3. Already Attempted Scenario: If views have already been provided but do not contain the correct information, attempt to find other views that may contain this information. 
       If there are no more views to select, return ['<NO_VIEWS>'].
    4. CONVERSATION HISTORY: When CONVERSATION HISTORY is provided, use it to resolve ambiguous references in the current QUESTION
       (e.g., "them", "those companies", "similar metrics") by understanding what entities, time periods, or metrics were discussed previously.

    Analyzes a natural language question and selects the most relevant Snowflake database views needed to answer the question.
    """
    
    # Input fields
    question: str = dspy.InputField(
        desc="User's natural language database query"
    )
    
    candidate_views: str = dspy.InputField(
        desc="List of available database views with descriptions, selectors, and columns formatted as a structured list"
    )
    
    conversation_history: str = dspy.InputField(
        desc="Previous conversation context for resolving references",
        default=""
    )
    
    domain_knowledge: str = dspy.InputField(
        desc="Financial domain rules (Asset Classes, Investment Classes, Platforms, Business Units)"
    )
    
    # Output fields
    reasoning: str = dspy.OutputField(
        desc="Step-by-step analysis of why specific views were selected"
    )
    
    selected_views: str = dspy.OutputField(
        desc="All views needed to answer the question, as a comma-separated list of selected view entity names, or '<NO_VIEWS>' if none are relevant"
    )


class SQLWriterSignature(dspy.Signature):
    """
    You are a specialized AI assistant focused EXCLUSIVELY on writing SQL queries for data retrieval from Snowflake databases.
    
    CRITICAL ROLE DEFINITION:
    - YOUR SOLE PURPOSE: Write SQL queries to retrieve data that will help answer the question
    - WHAT YOU MUST NOT DO: Provide final answers, interpretations, analysis, or conclusions
    - WHEN TO STOP: Once you have retrieved all necessary data, immediately return '<DONE>'
    
    Follow Chain of Thought:
    1. Understand the Question - identify intent, filters, aggregations, timeframes
    2. Map to Schema - identify relevant tables/columns, account for synonyms
    3. Construct SQL Query - ensure logic aligns with question intent
    
    NOTE: It is perfectly fine to return '<NO_QUERIES>' if queries cannot be written with given schemas.
    """
    
    # Input fields
    question: str = dspy.InputField(
        desc="User's natural language database query"
    )
    
    selected_view_schemas: str = dspy.InputField(
        desc="Detailed schema information for the selected database views"
    )
    
    conversation_history: str = dspy.InputField(
        desc="Previous conversation context for resolving ambiguous references",
        default=""
    )
    
    domain_rules: str = dspy.InputField(
        desc="Business glossary, financial term mappings, and SQL construction rules"
    )
    
    # Output fields
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning: (1) Question understanding, (2) Schema mapping, (3) Query construction logic"
    )
    
    sql_queries: str = dspy.OutputField(
        desc="Generated SQL query/queries, or '<NO_QUERIES>' if cannot be written, or '<DONE>' if all data retrieved"
    )