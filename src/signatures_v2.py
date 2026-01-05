import dspy
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

class EntityResolverSignature(dspy.Signature):
    """
    You are an entity resolver for MIC's Text2SQL system. Extract business entities from user queries.
    
    RULES:
    1. ONLY match terms explicitly mentioned in the query
    2. Match against: entity_mapper.json Values OR glossary.json Synonyms
    3. DO NOT infer or hallucinate entities not in the query
    4. For multi-type synonyms: Include in ALL matching entity types
    5. DO NOT include empty values - only successfully matched entities
    
    CRITICAL OUTPUT RULES - Entity Value Format:
    1. **ALWAYS RETURN THE EXACT VALUE FROM entity_mapper.json** - Never return the user's query text
    2. When you identify a match, return the EXACT entity value as it appears in entity_mapper.json values list
    3. Example: If user says "ADIC Direct Investments" and entity_mapper.json has "ADIC-Direct Investments", return "ADIC-Direct Investments"
    4. Example: If user says "private equity" and entity_mapper.json has "Private Equity", return "Private Equity" (exact capitalization)
    5. NEVER return partial values, symbols, JSON artifacts, punctuations or brackets
    6. NEVER return empty strings, whitespace only, or single characters
    7. If you cannot find a matching entity in the provided list, set that entity type to null or omit it entirely
    8. DO NOT extract or return JSON structure artifacts from the entity_mapper format
    9. DO NOT return the user's original query text - ONLY return normalized values from entity_mapper.json
    
    DISAMBIGUATION RULES: When a value exists in multiple entity types, use context clues:
    - If query contains "platform" → prefer PLATFORM over ASSET_CLASS
    - If query contains "business unit" or "unit" → prefer BUSINESS_UNIT
    - If query contains "asset class" or "asset type" → prefer ASSET_CLASS
    - If query contains "sector" → prefer GICS_SECTOR
    """
    
    # Inputs
    question: str = dspy.InputField(
        desc="User's natural language query"
    )
    
    entity_candidates: str = dspy.InputField(
        desc="""Available business entities from entity_mapper.json with exact values.
        Format: 
        - PLATFORM: [value1, value2, ...]
        - ASSET_CLASS: [value1, value2, ...]
        - BUSINESS_UNIT: [value1, value2, ...]
        
        Glossary (Abbreviation Expansions):
        - abbrev → expansion"""
    )
    
    disambiguation_hints: str = dspy.InputField(
        desc="Context clues for entity type disambiguation (e.g., 'platform' keyword suggests PLATFORM entity type)",
        default="No specific disambiguation hints"
    )
    
    # Outputs
    reasoning: str = dspy.OutputField(
        desc="""Step-by-step analysis:
        1. Identify terms in query that match entity candidates
        2. Apply disambiguation rules based on context keywords
        3. Match terms to EXACT values from entity_mapper.json
        4. Validate: no partial matches, no JSON artifacts, proper capitalization"""
    )
    
    resolved_entities: str = dspy.OutputField(
        desc="""Extracted entities as JSON string with EXACT values from entity_mapper.json.
        Examples:
        - {"PLATFORM": "Private Equity", "COUNTRY": "UAE"}
        - {"BUSINESS_UNIT": "ADIC-Direct Investments"}
        - {}  // If no entities found
        
        FORBIDDEN outputs: partial values, user's original text, JSON artifacts like "[", "]", ":", empty strings"""
    )

class ViewSelectorSignature(dspy.Signature):
    """
    You are an expert database view selector. Given a user query, you need to identify which database views contain the relevant data.
    
    IMPORTANT: The QUESTION may include RESOLVED ENTITIES section, which contains business entities that were extracted 
    from the original query by the entity resolver agent. Use these resolved entities to better understand the context 
    and select the most appropriate views.

    Key Instructions:
    1. Identify Relevant Views: Carefully analyze the QUESTION (including any RESOLVED ENTITIES if provided) and the 
       provided VIEW(s) descriptions, selectors and columns using your financial knowledge. And determine which view(s) 
       are most likely to contain the required data to answer the QUESTION.
    
    2. No Match Scenario: If none of the provided views appear relevant to the QUESTION, explicitly return '<NO_VIEWS>' 
       instead of suggesting irrelevant views. Remember, it's totally fine to say that it's not possible. If there are 
       no views to select, you must return '<NO_VIEWS>'.
    
    3. Already Attempted Scenario: If you have already provided views to the SQL Writer Agent but these do not contain 
       the correct information, attempt to find other views that may contain this information. If there are no more 
       views to select, you must return '<NO_VIEWS>'.
    
    4. CONVERSATION HISTORY: When CONVERSATION HISTORY is provided, use it to resolve ambiguous references in the 
       current QUESTION (e.g., "them", "those companies", "similar metrics") by understanding what entities, time 
       periods, or metrics were discussed previously.
    """
    
    # Inputs
    question: str = dspy.InputField(
        desc="User's natural language database query, may include RESOLVED ENTITIES section"
    )
    
    resolved_entities: str = dspy.InputField(
        desc="Entities extracted by Entity Resolver as JSON string: {'PLATFORM': 'Private Equity', ...}",
        default="{}"
    )
    
    candidate_views: str = dspy.InputField(
        desc="""Available database views with descriptions, selectors, and columns.
        Format:
        View: VIEW_NAME
        Entity: ENTITY_NAME
        Description: ...
        Selector: When to use this view
        Key Columns: col1, col2, ..."""
    )
    
    conversation_history: str = dspy.InputField(
        desc="Previous conversation context for resolving ambiguous references",
        default="No previous conversation"
    )
    
    domain_knowledge: str = dspy.InputField(
        desc="""Financial domain rules and view selection heuristics:
        - CRITICAL FINANCIAL CLASSIFICATION RULES
        - Entity type to view mapping
        - Special case handling (historical investments, active deals, etc.)"""
    )
    
    # Outputs
    reasoning: str = dspy.OutputField(
        desc="""Step-by-step view selection analysis:
        1. Analyze question keywords and resolved entities
        2. Match entities to appropriate view dimensions (PLATFORM → PLATFORM_KPI_VW)
        3. Consider special cases (historical investments, active deals, portfolio-level)
        4. Evaluate each candidate view's relevance
        5. Final decision with justification or '<NO_VIEWS>' if no match"""
    )
    
    selected_views: str = dspy.OutputField(
        desc="""Comma-separated list of selected view entity names (e.g., 'PLATFORM_KPI_VW, INVESTMENT_KPI_VW'), 
        or '<NO_VIEWS>' if none are relevant. Return ONLY entity names, not descriptions."""
    )