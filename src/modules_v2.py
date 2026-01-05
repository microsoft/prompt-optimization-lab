import dspy
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.signatures_v2 import EntityResolverSignature, ViewSelectorSignature

class EntityResolverModule(dspy.Module):
    """
    Entity Resolver using DSPy Chain-of-Thought.
    Implements logic from entity_resolver_agent.py and entity_resolver_template.py
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        
        # Chain of Thought for entity resolution
        self.entity_resolver_cot = dspy.ChainOfThought(EntityResolverSignature)
        
        # Load entity mapper and glossary
        self._load_entity_data()
        
    def _load_entity_data(self):
        """Load entity_mapper.json and glossary.json"""
        try:
            # Find project root
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir
            while not (project_root / "data").exists() and project_root != project_root.parent:
                project_root = project_root.parent
            
            # Load entity mapper
            entity_mapper_path = project_root / "data/assets/entity_mapper.json"
            with open(entity_mapper_path, 'r', encoding='utf-8') as f:
                entity_mapper_data = json.load(f)
            
            # Load glossary
            glossary_path = project_root / "data/assets/glossary.json"
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_data = json.load(f)
            
            # Format entity candidates
            self.entity_candidates = self._format_entity_candidates(
                entity_mapper_data, 
                glossary_data
            )
            
        except Exception as e:
            print(f"Error loading entity data: {e}")
            self.entity_candidates = "No entity data available"
    
    def _format_entity_candidates(self, entity_mapper: dict, glossary: dict) -> str:
        """Format entity candidates for prompt (matching EntityResolverAgent logic)"""
        entity_list = []
        
        # Format entity mapper values
        if 'entities' in entity_mapper:
            for item in entity_mapper['entities']:
                entity_name = item.get('entity', '')
                values = item.get('values', [])
                
                entity_info = f"- {entity_name}:\n"
                entity_info += f"  Values: {', '.join(map(str, values))}\n"
                entity_list.append(entity_info)
        
        # Format glossary (abbreviations)
        glossary_info = "\n\n## Glossary (Abbreviation Expansions):\n"
        if 'acronyms' in glossary:
            for abbrev, expansion in glossary['acronyms'].items():
                glossary_info += f"- {abbrev} → {expansion}\n"
        
        return '\n'.join(entity_list) + glossary_info
    
    def _generate_disambiguation_hints(self, question: str) -> str:
        """Generate contextual hints (matching EntityResolverAgent._generate_disambiguation_hints)"""
        question_lower = question.lower()
        hints = []
        
        if "platform" in question_lower:
            hints.append("CONTEXT HINT: The query mentions 'platform' - prefer PLATFORM entity type over ASSET_CLASS when ambiguous")
        if "business unit" in question_lower or " unit" in question_lower:
            hints.append("CONTEXT HINT: The query mentions 'unit' - prefer BUSINESS_UNIT entity type when ambiguous")
        if "asset class" in question_lower or "asset type" in question_lower:
            hints.append("CONTEXT HINT: The query mentions 'asset class/type' - prefer ASSET_CLASS entity type when ambiguous")
        if "sector" in question_lower:
            hints.append("CONTEXT HINT: The query mentions 'sector' - prefer GICS_SECTOR entity type when ambiguous")
        
        return "\n".join(hints) if hints else "No specific disambiguation hints"
    
    def _validate_and_clean_entities(self, entities_str: str) -> str:
        """Validate entities (matching EntityResolverAgent validation logic)"""
        try:
            entities = json.loads(entities_str)
            
            if not isinstance(entities, dict):
                return "{}"
            
            cleaned_entities = {}
            
            for entity_type, entity_value in entities.items():
                # Check if value exists and is string
                if not entity_value or not isinstance(entity_value, str):
                    continue
                
                # Clean whitespace
                entity_value = entity_value.strip()
                
                # Minimum length check
                if len(entity_value) < 2:
                    continue
                
                # Valid characters pattern
                valid_pattern = r'^[\w\s\-&,.\'"()]+$'
                if not re.match(valid_pattern, entity_value):
                    continue
                
                # Check for JSON artifacts
                json_artifacts = [':[', ']:', '[', ']', '{', '}', '::']
                if any(artifact in entity_value for artifact in json_artifacts):
                    continue
                
                # Passed all validations
                cleaned_entities[entity_type] = entity_value
            
            return json.dumps(cleaned_entities) if cleaned_entities else "{}"
            
        except json.JSONDecodeError:
            return "{}"
    
    def forward(self, question: str) -> dspy.Prediction:
        """
        Extract and normalize business entities from user query.
        
        Args:
            question: User's natural language query
            
        Returns:
            dspy.Prediction with reasoning and resolved_entities
        """
        # Generate disambiguation hints
        disambiguation_hints = self._generate_disambiguation_hints(question)
        
        # Run entity resolution with CoT
        result = self.entity_resolver_cot(
            question=question,
            entity_candidates=self.entity_candidates,
            disambiguation_hints=disambiguation_hints
        )
        
        # Validate and clean entities
        validated_entities = self._validate_and_clean_entities(result.resolved_entities)
        
        # Return prediction with validated entities
        return dspy.Prediction(
            reasoning=result.reasoning,
            resolved_entities=validated_entities
        )


class ViewSelectorModule(dspy.Module):
    """
    View Selector using DSPy Chain-of-Thought.
    Implements logic from view_selector_agent.py and view_selector_template.py
    """
    
    def __init__(self, config: dict, candidate_views: List[Dict] = None):
        super().__init__()
        self._config = config
        
        # Chain of Thought for view selection
        self.view_selector_cot = dspy.ChainOfThought(ViewSelectorSignature)
        
        # Store candidate views
        self.candidate_views = candidate_views or []
        self._formatted_views = self._format_candidate_views()
        
        # Load domain knowledge from config and examples
        self._load_domain_knowledge()
    
    def _load_domain_knowledge(self):
        """Load domain knowledge from view_selector_examples.md"""
        try:
            # Find project root
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir
            while not (project_root / "data").exists() and project_root != project_root.parent:
                project_root = project_root.parent
            
            # Load view selector examples
            examples_path = project_root / "data/assets/view_selector_examples.md"
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples_content = f.read()
            
            # Combine with hardcoded rules
            self.domain_knowledge = f"""{examples_content}"""
        except Exception as e:
            print(f"Error loading domain knowledge: {e}")
            self.domain_knowledge = "Default domain knowledge"
    
    def _format_candidate_views(self) -> str:
        """Format candidate views for prompt (matching ViewSelectorAgent logic)"""
        if not self.candidate_views:
            return "No views available"
        
        formatted = []
        for view in self.candidate_views:
            view_str = f"View: {view.get('view_name', 'Unknown')}\n"
            view_str += f"Entity: {view.get('entity', 'Unknown')}\n"
            view_str += f"Description: {view.get('description', 'N/A')}\n"
            view_str += f"Selector: {view.get('selector', 'N/A')}\n"
            
            # # Add column info if available
            # if 'columns' in view and view['columns']:
            #     cols = ', '.join(view['columns'][:15])  # First 15 columns
            #     if len(view['columns']) > 15:
            #         cols += f", ... ({len(view['columns'])} total columns)"
            #     view_str += f"Key Columns: {cols}\n"
            
            formatted.append(view_str)
        
        return "\n---\n".join(formatted)
    
    def forward(
        self, 
        question: str, 
        resolved_entities: str = "{}",
        conversation_history: str = ""
    ) -> dspy.Prediction:
        """
        Select the most relevant database views.
        
        Args:
            question: User's database query
            resolved_entities: JSON string of resolved entities
            conversation_history: Previous conversation context
            
        Returns:
            dspy.Prediction with reasoning and selected_views
        """
        # Enhanced question with resolved entities (matching context_gen_system.py logic)
        enhanced_question = question
        if resolved_entities and resolved_entities.strip() != "{}":
            try:
                entities_dict = json.loads(resolved_entities)
                if entities_dict:
                    entities_text = "\n\nRESOLVED ENTITIES:\n" + "\n".join(
                        f"{entity_type}: {entity_value}" 
                        for entity_type, entity_value in entities_dict.items()
                    )
                    enhanced_question += entities_text
            except json.JSONDecodeError:
                pass
        
        # Run view selection with CoT
        result = self.view_selector_cot(
            question=enhanced_question,
            resolved_entities=resolved_entities,
            candidate_views=self._formatted_views,
            conversation_history=conversation_history or "No previous conversation",
            domain_knowledge=self.domain_knowledge
        )
        
        return result


class IntegratedText2SQLModule(dspy.Module):
    """
    Integrated two-stage module: Entity Resolution → View Selection
    Matches the exact flow from context_gen_system.py
    """
    
    def __init__(self, config: dict, candidate_views: List[Dict] = None):
        super().__init__()
        self._config = config
        
        # Stage 1: Entity Resolution
        self.entity_resolver = EntityResolverModule(config)
        
        # Stage 2: View Selection
        self.view_selector = ViewSelectorModule(config, candidate_views)
    
    def forward(
        self, 
        question: str, 
        conversation_history: str = ""
    ) -> dspy.Prediction:
        """
        Two-stage reasoning: Entity Resolution → View Selection
        (Matches context_gen_system.py generate_response flow)
        
        Args:
            question: User's database query
            conversation_history: Previous conversation context
            
        Returns:
            dspy.Prediction with all intermediate and final results
        """
        # Stage 1: Entity Resolution
        entity_result = self.entity_resolver(question=question)
        
        resolved_entities_str = entity_result.resolved_entities
        entity_reasoning = entity_result.reasoning
        
        # Stage 2: View Selection (using resolved entities)
        view_result = self.view_selector(
            question=question,
            resolved_entities=resolved_entities_str,
            conversation_history=conversation_history
        )
        
        selected_views_str = view_result.selected_views
        view_reasoning = view_result.reasoning
        
        # Combined reasoning trace
        combined_reasoning = f"""
=== STAGE 1: ENTITY RESOLUTION ===
{entity_reasoning}

Resolved Entities: {resolved_entities_str}

=== STAGE 2: VIEW SELECTION ===
{view_reasoning}

Selected Views: {selected_views_str}
"""
        
        # Return comprehensive prediction
        return dspy.Prediction(
            # Stage 1 outputs
            entity_reasoning=entity_reasoning,
            resolved_entities=resolved_entities_str,
            
            # Stage 2 outputs
            view_reasoning=view_reasoning,
            selected_views=selected_views_str,
            
            # Combined
            combined_reasoning=combined_reasoning.strip(),
            reasoning=combined_reasoning.strip()  # Alias for compatibility
        )
