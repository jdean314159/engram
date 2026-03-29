"""
Project-Specific Helper Functions for Semantic Memory

Convenience methods for common operations in each project.

Author: Jeffrey Dean
"""

import time
from typing import List, Dict, Any, Optional
from .semantic_memory import SemanticMemory, ProjectType


class ProgrammingAssistantHelpers:
    """Helper functions for programming assistant."""
    
    def __init__(self, memory: SemanticMemory):
        self.memory = memory
    
    def add_code_dependency(
        self,
        from_snippet_id: str,
        to_snippet_id: str,
        dependency_type: str = "imports",
        strength: float = 0.8
    ):
        """Add dependency between code snippets."""
        return self.memory.add_relationship(
            "CodeSnippet", from_snippet_id,
            "CodeSnippet", to_snippet_id,
            "DEPENDS_ON",
            {"dependency_type": dependency_type, "strength": strength}
        )
    
    def find_learning_path(self, target_concept: str) -> List[Dict[str, Any]]:
        """Find shortest learning path to target concept."""
        cypher = """
            MATCH path = shortestPath((start:Concept)-[:REQUIRES*]->(target:Concept))
            WHERE target.name = $target_name
            RETURN [node in nodes(path) | node.name] as path_names,
                   [node in nodes(path) | node.difficulty] as difficulties,
                   length(path) as steps
            ORDER BY steps
            LIMIT 1
        """
        return self.memory.query(cypher, {"target_name": target_concept})
    
    def find_prerequisites(self, concept_name: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find all prerequisites for a concept."""
        cypher = f"""
            MATCH (c:Concept)-[:REQUIRES*1..{max_depth}]->(prereq:Concept)
            WHERE c.name = $concept_name
            RETURN prereq.name as name, prereq.difficulty as difficulty,
                   prereq.category as category
        """
        return self.memory.query(cypher, {"concept_name": concept_name})
    
    def find_similar_bugs(self, error_message: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find bugs with similar error messages."""
        cypher = """
            MATCH (b:Bug)
            WHERE b.message CONTAINS $keyword
            RETURN b.error_type as type, b.message as message,
                   b.solution as solution, b.frequency as frequency
            ORDER BY b.frequency DESC
            LIMIT $limit
        """
        # Extract key terms from error message
        keywords = error_message.split()[:3]  # Simplified
        keyword = keywords[0] if keywords else ""
        
        return self.memory.query(cypher, {"keyword": keyword, "limit": limit})
    
    def find_alternatives(self, snippet_id: str) -> List[Dict[str, Any]]:
        """Find alternative implementations."""
        cypher = """
            MATCH (s:CodeSnippet)-[alt:ALTERNATIVE_TO]->(other:CodeSnippet)
            WHERE s.id = $snippet_id
            RETURN other.code as code, other.description as description,
                   alt.performance_diff as performance_diff, alt.notes as notes
        """
        return self.memory.query(cypher, {"snippet_id": snippet_id})


class FileOrganizerHelpers:
    """Helper functions for file organizer (90K photos)."""
    
    def __init__(self, memory: SemanticMemory):
        self.memory = memory
    
    def add_photo(
        self,
        photo_id: str,
        path: str,
        timestamp: float,
        metadata: Dict[str, Any]
    ):
        """Add photo node with metadata."""
        props = {
            "path": path,
            "filename": path.split("/")[-1],
            "timestamp": timestamp,
            "size": metadata.get("size", 0),
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "hash": metadata.get("hash", ""),
            "camera": metadata.get("camera", ""),
            "metadata": str(metadata),
        }
        return self.memory.add_node("Photo", photo_id, props)
    
    def detect_events(
        self,
        time_window_hours: float = 1.0,
        location_threshold_km: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect events: photos taken within time/location window."""
        cypher = """
            MATCH (p1:Photo)-[:TAKEN_AT]->(loc:Location)<-[:TAKEN_AT]-(p2:Photo)
            WHERE abs(p1.timestamp - p2.timestamp) < $time_window
              AND p1.id < p2.id
            RETURN p1.path as photo1, p2.path as photo2,
                   loc.name as location, abs(p1.timestamp - p2.timestamp) as time_diff
            ORDER BY time_diff
        """
        time_window_sec = time_window_hours * 3600
        return self.memory.query(cypher, {"time_window": time_window_sec})
    
    def find_duplicates(self, similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Find duplicate photos based on similarity."""
        cypher = """
            MATCH (p1:Photo)-[sim:SIMILAR_TO]->(p2:Photo)
            WHERE sim.similarity >= $threshold AND p1.id < p2.id
            RETURN p1.path as photo1, p2.path as photo2, sim.similarity as similarity
            ORDER BY sim.similarity DESC
        """
        return self.memory.query(cypher, {"threshold": similarity_threshold})
    
    def find_photos_by_person(self, person_name: str) -> List[Dict[str, Any]]:
        """Find all photos containing a specific person."""
        cypher = """
            MATCH (photo:Photo)-[:CONTAINS]->(person:Person)
            WHERE person.name = $name
            RETURN photo.path as path, photo.timestamp as timestamp
            ORDER BY photo.timestamp DESC
        """
        return self.memory.query(cypher, {"name": person_name})
    
    def find_photos_by_location(self, country: str) -> List[Dict[str, Any]]:
        """Find photos taken in a specific country."""
        cypher = """
            MATCH (photo:Photo)-[:TAKEN_AT]->(loc:Location)
            WHERE loc.country = $country
            RETURN photo.path as path, loc.city as city, photo.timestamp as timestamp
            ORDER BY photo.timestamp DESC
        """
        return self.memory.query(cypher, {"country": country})
    
    def cluster_by_event(self) -> List[Dict[str, Any]]:
        """Group photos by detected events."""
        cypher = """
            MATCH (event:Event)<-[:PART_OF]-(photo:Photo)
            RETURN event.name as event, count(photo) as photo_count,
                   min(photo.timestamp) as start_time, max(photo.timestamp) as end_time
            ORDER BY start_time DESC
        """
        return self.memory.query(cypher)


class LanguageTutorHelpers:
    """Helper functions for language tutor."""
    
    def __init__(self, memory: SemanticMemory):
        self.memory = memory
    
    def add_vocabulary(
        self,
        word_id: str,
        word: str,
        translation: str,
        language: str = "spanish",
        difficulty: str = "beginner"
    ):
        """Add vocabulary word."""
        props = {
            "word": word,
            "language": language,
            "translation": translation,
            "difficulty": difficulty,
            "part_of_speech": "",
            "examples": "[]",
        }
        return self.memory.add_node("VocabularyWord", word_id, props)
    
    def track_mastery(
        self,
        user_id: str,
        word_id: str,
        mastery_level: float
    ):
        """Track user's mastery of a word."""
        return self.memory.add_relationship(
            "User", user_id,
            "VocabularyWord", word_id,
            "MASTERED",
            {"mastery_level": mastery_level, "last_practiced": time.time()}
        )
    
    def find_unmastered_words(
        self,
        user_id: str,
        difficulty: str = "beginner"
    ) -> List[Dict[str, Any]]:
        """Find words user hasn't mastered yet."""
        cypher = """
            MATCH (w:VocabularyWord)
            WHERE w.difficulty = $difficulty
              AND NOT EXISTS {
                MATCH (u:User)-[:MASTERED]->(w)
                WHERE u.id = $user_id
              }
            RETURN w.word as word, w.translation as translation
            LIMIT 20
        """
        return self.memory.query(cypher, {"user_id": user_id, "difficulty": difficulty})
    
    def find_confused_pairs(self, word_id: str) -> List[Dict[str, Any]]:
        """Find words commonly confused with given word."""
        cypher = """
            MATCH (w1:VocabularyWord)-[conf:CONFUSED_WITH]->(w2:VocabularyWord)
            WHERE w1.id = $word_id
            RETURN w2.word as word, w2.translation as translation, conf.frequency as frequency
            ORDER BY conf.frequency DESC
        """
        return self.memory.query(cypher, {"word_id": word_id})
    
    def find_learning_path_for_concept(
        self,
        target_concept: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find prerequisite path for language concept."""
        cypher = f"""
            MATCH path = (start:LanguageConcept)-[:REQUIRES_CONCEPT*1..{max_depth}]->(target:LanguageConcept)
            WHERE target.name = $target_name
            RETURN [node in nodes(path) | node.name] as path,
                   length(path) as steps
            ORDER BY steps
            LIMIT 1
        """
        return self.memory.query(cypher, {"target_name": target_concept})


class VoiceInterfaceHelpers:
    """Helper functions for voice interface."""
    
    def __init__(self, memory: SemanticMemory):
        self.memory = memory
    
    def add_command(
        self,
        command_id: str,
        text: str,
        intent: str,
        entities: Dict[str, Any]
    ):
        """Add voice command."""
        props = {
            "text": text,
            "intent": intent,
            "entities": str(entities),
            "frequency": 1,
            "last_used": time.time(),
        }
        return self.memory.add_node("Command", command_id, props)
    
    def track_command_sequence(
        self,
        from_command_id: str,
        to_command_id: str,
        delay_seconds: float
    ):
        """Track command sequences (X follows Y)."""
        # Get existing relationship
        cypher = """
            MATCH (a:Command {id: $from_id})-[f:FOLLOWS]->(b:Command {id: $to_id})
            RETURN f.frequency as freq, f.avg_delay as delay
        """
        results = self.memory.query(cypher, {"from_id": from_command_id, "to_id": to_command_id})
        
        if results:
            # Update existing
            old_freq = results[0]["freq"]
            old_delay = results[0]["delay"]
            new_freq = old_freq + 1
            new_delay = (old_delay * old_freq + delay_seconds) / new_freq
            
            # Delete and recreate with new values
            cypher_del = """
                MATCH (a:Command {id: $from_id})-[f:FOLLOWS]->(b:Command {id: $to_id})
                DELETE f
            """
            self.memory.conn.execute(cypher_del, {"from_id": from_command_id, "to_id": to_command_id})
            
            return self.memory.add_relationship(
                "Command", from_command_id,
                "Command", to_command_id,
                "FOLLOWS",
                {"frequency": new_freq, "avg_delay": new_delay}
            )
        else:
            # Create new
            return self.memory.add_relationship(
                "Command", from_command_id,
                "Command", to_command_id,
                "FOLLOWS",
                {"frequency": 1, "avg_delay": delay_seconds}
            )
    
    def predict_next_command(
        self,
        current_command_id: str,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Predict likely next commands."""
        cypher = """
            MATCH (current:Command {id: $command_id})-[f:FOLLOWS]->(next:Command)
            RETURN next.text as command, f.frequency as frequency, f.avg_delay as avg_delay
            ORDER BY f.frequency DESC
            LIMIT $limit
        """
        return self.memory.query(cypher, {"command_id": current_command_id, "limit": top_n})
