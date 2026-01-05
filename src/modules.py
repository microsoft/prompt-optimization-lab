"""
DSPy Modules for View Selector and SQL Writer agents.
"""

import dspy
from typing import List, Dict, Optional
from .signatures import ViewSelectorSignature, SQLWriterSignature


class ViewSelectorModule(dspy.Module):
    """
    Baseline database view selector using DSPy Chain-of-Thought reasoning.
    
    This module implements the core logic:
    1. Analyze the question and available views
    2. Apply financial/domain knowledge
    3. Handle conversation history for context
    4. Return list of appropriate views or <NO_VIEWS>
    """
    
    def __init__(self, candidate_views: List[Dict] = None):
        super().__init__()
        
        # Use Chain of Thought for step-by-step reasoning
        self.selector_cot = dspy.ChainOfThought(ViewSelectorSignature)
        
        # Embed domain knowledge
        self.domain_knowledge = """
CRITICAL FINANCIAL CLASSIFICATION RULES:
   - For questions related to historical investments, always use the **INVESTMENT_KPI_VW**.
   - To retrieve Partner and active deals information, ALWAYS USE the **ACTIVE_DEAL_LIST_VW**
   - "PE" typically refers to "Private Equity"
   - Country/geography questions need location-aware views
   - Exposure questions need portfolio/position views
   - Fund-level questions need fund detail views
        """
        
        # Store candidate views
        self.candidate_views = candidate_views or []
        self._formatted_views = self._format_candidate_views()
    
    def _format_candidate_views(self) -> str:
        """Format candidate views for prompt."""
        if not self.candidate_views:
            return ""
        
        formatted = []
        for view in self.candidate_views:
            view_str = f"View: {view.get('view_name', 'Unknown')}\n"
            view_str += f"Entity: {view.get('entity', 'Unknown')}\n"
            view_str += f"Description: {view.get('description', 'N/A')}\n"
            view_str += f"Selector: {view.get('selector', 'N/A')}\n"
            formatted.append(view_str)
        
        return "\n---\n".join(formatted)
    
    def forward(self, question: str, conversation_history: str = ""):
        """
        Select the most relevant database views for the given question.
        
        Args:
            question: User's database query
            conversation_history: Previous conversation context
            
        Returns:
            dspy.Prediction with reasoning and selected_views
        """
        # Use Chain-of-Thought reasoning to select views
        cot_result = self.selector_cot(
            question=question,
            candidate_views=self._formatted_views,
            conversation_history=conversation_history or "No previous conversation",
            domain_knowledge=self.domain_knowledge
        )
        # Step 5: Return DSPy Prediction object (NOT dictionary)
        # This is critical for GEPA optimizer to work properly
        return cot_result  # Return the Prediction object directly


class SQLWriterModule(dspy.Module):
    """SQL Writer module that generates Snowflake queries from selected view schemas."""
    
    def __init__(self):
        super().__init__()
        self.sql_writer_cot = dspy.ChainOfThought(SQLWriterSignature)
        
        # Embed domain rules
        self.domain_rules = """
BUSINESS GLOSSARY - Financial Terms to SQL Mapping:
        - "largest investments", "top investments": UNREALIZED_VALUE
        - "largest deployments", "biggest cash investments": ITD_CASH_INVESTED
        - "exposure", "current exposure": UNREALIZED_VALUE
        - "IC Case IRR", "target IRR": TARGET_IRR / TARGET_RETURN
        - "IC Case MOIC", "target MOIC": TARGET_MOIC
        
        CRITICAL RULES:
        A. Schema Usage:
           - Only use provided schema/columns, never guess or makeup names
           - Focus on column's "name", "definition", "type", "allowed_values"
           - Use CONVERSATION HISTORY to resolve ambiguous references
        
        B. Deal-Specific Rules:
           - Use DEAL_ID from DEAL METADATA unless explicitly asking about another deal
           - ALWAYS include descriptive text values alongside IDs (DealName + DealId)
        
        C. SQL Construction (Snowflake):
           - Ensure NULL values at bottom: ORDER BY <col> DESC NULLS LAST
           - NO data modification queries (INSERT, UPDATE, DELETE, DROP, etc.)
           - NO sensitive info exposure (credentials, passwords, schemas)
           - ALWAYS include DATE/TIME fields in SELECT
           - **MANDATORY FUZZY MATCHING**: Use ILIKE '%term%' for text searches
           - Prioritize AED currency columns over USD unless specified
        
        D. Domain-Specific Rules:
           - Comparison/Ranking: Fetch ALL relevant values, let user derive answer
           - GICS Sector: Use "GICS_SECTOR", NOT "ASSET_CLASS_NAME"
           - **NEGATIVE VALUES**: Use ABS() for sorting (ORDER BY ABS(col) DESC)
           - **LATEST DATA**: Default to most recent records unless date specified
           - CRITICAL: NEVER use RECORD_TYPE column for filtering
        
        E. Timeseries Data Rules:
           - CURRENT YEAR: Use current calendar year
           - LAST YEAR: Use (current year - 1)
           - DEFAULT: If no year specified, use current year
           - Filter timestamped views by appropriate date ranges
        
        F. Historical Investments:
           - Use DEAL METADATA for Partner name
           - Use INVESTMENT_KPI_VW for historical investments
           - Use partial matching: WHERE LENGTH('%PARTNERS%') > 2 AND PARTNERS LIKE '%PARTNERS%'
        
        G. Calculations:
           - Growth Rate/CAGR: ((End/Start)^(1/Years)) - 1
           - Only use UNREALIZED_VALUE for growth calculations
        
        H. Performance:
           - Select only necessary columns
           - Push filters early, avoid wildcard selects
           - Row limits automatically applied
        
        FINAL: Return '<NO_QUERIES>' if information not available, '<DONE>' when all data retrieved.
        """
    
    def forward(self, question: str, selected_view_schemas: str, 
                conversation_history: str = ""):
        """Generate SQL queries for the given question using selected view schemas."""
        
        result = self.sql_writer_cot(
            question=question,
            selected_view_schemas=selected_view_schemas,
            conversation_history=conversation_history or "No previous conversation",
            domain_rules=self.domain_rules
        )
        
        return result
    
class ViewToSQLPipeline(dspy.Module):
    """
    Integrated pipeline: View Selection → Schema Retrieval → SQL Generation
    
    This module chains the ViewSelector and SQLWriter together to produce
    complete SQL queries from natural language questions.
    """
    
    def __init__(self, candidate_views: List[Dict], schema_retriever_func=None):
        """
        Initialize the pipeline.
        
        Args:
            candidate_views: List of available database views with metadata
            schema_retriever_func: Optional function to retrieve detailed schemas.
                                  If None, uses basic view information.
        """
        super().__init__()
        
        # Initialize both agents
        self.view_selector = ViewSelectorModule(candidate_views=candidate_views)
        self.sql_writer = SQLWriterModule()
        
        # Store schema retrieval function
        self.schema_retriever = schema_retriever_func or self._default_schema_retriever
        self.candidate_views = candidate_views
    
    def _default_schema_retriever(self, selected_view_names: List[str]) -> str:
        """
        Default schema retriever that formats view schemas from candidate_views.
        
        Args:
            selected_view_names: List of view names/entities to retrieve schemas for
            
        Returns:
            Formatted schema information string
        """
        schemas = []
        
        for view_name in selected_view_names:
            # Find the view in candidate_views (match by entity or view_name)
            matching_view = None
            for view in self.candidate_views:
                if (view.get('entity', '').lower() == view_name.lower() or 
                    view.get('view_name', '').lower() == view_name.lower()):
                    matching_view = view
                    break
            
            if matching_view:
                schema_str = f"Schema for {matching_view.get('view_name', 'Unknown')} (Entity: {matching_view.get('entity', 'Unknown')}):\n"
                schema_str += f"Description: {matching_view.get('description', 'N/A')}\n"
                schema_str += f"Selector: {matching_view.get('selector', 'N/A')}\n\n"
                schema_str += "Columns:\n"
                
                # Add column information
                columns = matching_view.get('columns', [])
                for col in columns:
                    schema_str += f"- {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} - {col.get('definition', 'N/A')}\n"
                    
                    # Add allowed values if present
                    if 'allowed_values' in col and col['allowed_values']:
                        schema_str += f"   Allowed Values: {col['allowed_values']}\n"
                    
                    # Add sample values if present
                    if 'sample_values' in col and col['sample_values']:
                        schema_str += f"   Sample Values: {col['sample_values']}\n"
                    
                    schema_str += "\n"
                
                schemas.append(schema_str)
        
        return "\n\n".join(schemas)
    
    def _parse_selected_views(self, selected_views_str: str) -> List[str]:
        """
        Parse the selected views string into a list of view names.
        
        Args:
            selected_views_str: Comma-separated view names or '<NO_VIEWS>'
            
        Returns:
            List of view names
        """
        if not selected_views_str or '<NO_VIEWS>' in selected_views_str:
            return []
        
        # Handle various separators
        views = selected_views_str.replace('\n', ',').replace(';', ',').split(',')
        return [v.strip() for v in views if v.strip()]
    
    def forward(self, question: str, conversation_history: str = "", 
                deal_metadata: str = "", max_iterations: int = 3):
        """
        Execute the complete pipeline: View Selection → SQL Generation
        
        Args:
            question: User's natural language database query
            conversation_history: Previous conversation context
            deal_metadata: Current deal context (DEAL_ID, Partner, Region, etc.)
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            dspy.Prediction with view_selection, sql_generation, and metadata
        """
        
        iteration = 0
        all_queries = []
        attempted_views = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # Step 1: Select relevant views
            print(f"\n🔍 Iteration {iteration}: Selecting views...")
            view_selection = self.view_selector(
                question=question,
                conversation_history=conversation_history
            )
            
            selected_views_str = view_selection.selected_views
            selected_views = self._parse_selected_views(selected_views_str)
            
            # Handle no views case
            if not selected_views or '<NO_VIEWS>' in selected_views_str:
                print("❌ No relevant views found")
                return dspy.Prediction(
                    view_selection=view_selection,
                    sql_queries='<NO_QUERIES>',
                    reasoning=f"No relevant views found. {view_selection.reasoning}",
                    selected_views=selected_views_str,
                    iterations=iteration
                )
            
            # Step 2: Retrieve detailed schemas for selected views
            print(f"📊 Retrieved views: {', '.join(selected_views)}")
            schemas = self.schema_retriever(selected_views)
            attempted_views.extend(selected_views)
            
            # Step 3: Generate SQL queries
            print("✍️ Generating SQL queries...")
            sql_result = self.sql_writer(
                question=question,
                selected_view_schemas=schemas,
                conversation_history=conversation_history
            )
            
            # Check if done or no queries
            if '<DONE>' in sql_result.sql_queries or '<NO_QUERIES>' in sql_result.sql_queries:
                print(f"✅ Pipeline complete: {sql_result.sql_queries[:50]}...")
                return dspy.Prediction(
                    view_selection=view_selection,
                    sql_queries=sql_result.sql_queries,
                    reasoning=sql_result.reasoning,
                    selected_views=selected_views_str,
                    view_reasoning=view_selection.reasoning,
                    iterations=iteration,
                    all_attempted_views=list(set(attempted_views))
                )
            
            # Collect queries
            all_queries.append({
                'iteration': iteration,
                'views': selected_views,
                'queries': sql_result.sql_queries,
                'reasoning': sql_result.reasoning
            })
            
            # For now, return after first successful query generation
            # In production, you might execute and refine based on results
            return dspy.Prediction(
                view_selection=view_selection,
                sql_queries=sql_result.sql_queries,
                reasoning=sql_result.reasoning,
                selected_views=selected_views_str,
                view_reasoning=view_selection.reasoning,
                iterations=iteration,
                all_queries=all_queries,
                all_attempted_views=list(set(attempted_views))
            )
        
        # Max iterations reached
        print(f"⚠️ Max iterations ({max_iterations}) reached")
        return dspy.Prediction(
            view_selection=view_selection if 'view_selection' in locals() else None,
            sql_queries=sql_result.sql_queries if 'sql_result' in locals() else '<NO_QUERIES>',
            reasoning="Max iterations reached without completion",
            selected_views=selected_views_str if 'selected_views_str' in locals() else '<NO_VIEWS>',
            iterations=iteration,
            warning="Max iterations reached"
        )
    