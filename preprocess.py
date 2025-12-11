"""
Preprocessing Module
Enriches raw LinkedIn posts with metadata (line count, language, tags)
and unifies tags to a standardized list.
"""
import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_helper import get_llm

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_POSTS_PATH = os.path.join(DATA_DIR, "raw_posts.json")
PROCESSED_POSTS_PATH = os.path.join(DATA_DIR, "processed_posts.json")


def load_raw_posts():
    """Load raw posts from JSON file."""
    with open(RAW_POSTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def count_lines(text: str) -> int:
    """Count non-empty lines in text."""
    lines = [line for line in text.split("\n") if line.strip()]
    return len(lines)


def extract_metadata_prompt():
    """Create prompt template for metadata extraction."""
    template = """Analyze the following LinkedIn post and extract metadata.

POST:
{post_text}

Return a JSON object with the following fields:
- "language": Either "English" or "Hinglish" (Hindi+English mix)
- "tags": A list of 2-4 relevant topic tags (e.g., "Causal AI", "Machine Learning", "Financial Engineering", "System Design", "Quantitative Finance", "Statistics", "Trading", "Data Engineering")

Return ONLY the JSON object, no additional text.

Example output:
{{"language": "English", "tags": ["Machine Learning", "Causal AI"]}}

JSON Output:"""
    
    return PromptTemplate(
        input_variables=["post_text"],
        template=template
    )


def unify_tags_prompt():
    """Create prompt template for tag unification."""
    template = """You are given a list of tags extracted from LinkedIn posts. 
Many tags are similar but worded differently.

ALL EXTRACTED TAGS:
{all_tags}

Create a mapping that standardizes these tags into a unified list. 
Group similar concepts together and pick the best representative name.

Return a JSON object where:
- Keys are the original tags
- Values are the standardized tag names

Use these preferred standardized categories when applicable:
- "Causal AI" (for causal inference, causal ML, etc.)
- "Machine Learning" (for ML, predictive modeling, etc.)
- "Financial Engineering" (for derivatives, options, volatility, etc.)
- "Quantitative Finance" (for quant, trading strategies, alpha, etc.)
- "System Design" (for architecture, data pipelines, etc.)
- "Data Engineering" (for data processing, databases, etc.)
- "Statistics" (for econometrics, statistical methods, etc.)
- "Programming" (for Python, C++, Rust, coding, etc.)

Return ONLY the JSON mapping object.

JSON Output:"""
    
    return PromptTemplate(
        input_variables=["all_tags"],
        template=template
    )


def extract_metadata_for_post(llm, post_text: str) -> dict:
    """Extract metadata for a single post using LLM."""
    prompt = extract_metadata_prompt()
    parser = JsonOutputParser()
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"post_text": post_text})
        return result
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        # Fallback to basic metadata
        return {
            "language": "English",
            "tags": ["General"]
        }


def unify_all_tags(llm, all_tags: list) -> dict:
    """Create a mapping to unify all tags to standardized names."""
    prompt = unify_tags_prompt()
    parser = JsonOutputParser()
    
    chain = prompt | llm | parser
    
    unique_tags = list(set(all_tags))
    
    try:
        result = chain.invoke({"all_tags": json.dumps(unique_tags)})
        return result
    except Exception as e:
        print(f"Error unifying tags: {e}")
        # Return identity mapping as fallback
        return {tag: tag for tag in unique_tags}


def preprocess_posts():
    """Main preprocessing pipeline."""
    print("Loading raw posts...")
    raw_posts = load_raw_posts()
    print(f"Loaded {len(raw_posts)} posts")
    
    print("\nInitializing LLM...")
    llm = get_llm()
    
    # Step 1: Extract metadata for each post
    print("\n--- Step 1: Extracting metadata ---")
    all_tags = []
    enriched_posts = []
    
    for i, post in enumerate(raw_posts):
        print(f"Processing post {i+1}/{len(raw_posts)}...")
        
        text = post["text"]
        line_count = count_lines(text)
        
        # Get metadata from LLM
        metadata = extract_metadata_for_post(llm, text)
        
        enriched_post = {
            "text": text,
            "engagement": post.get("engagement", 0),
            "line_count": line_count,
            "language": metadata.get("language", "English"),
            "tags": metadata.get("tags", [])
        }
        
        enriched_posts.append(enriched_post)
        all_tags.extend(enriched_post["tags"])
        
        print(f"  Lines: {line_count}, Language: {enriched_post['language']}, Tags: {enriched_post['tags']}")
    
    # Step 2: Unify tags
    print("\n--- Step 2: Unifying tags ---")
    print(f"Found {len(set(all_tags))} unique tags before unification")
    
    tag_mapping = unify_all_tags(llm, all_tags)
    print(f"Tag mapping created: {json.dumps(tag_mapping, indent=2)}")
    
    # Apply unified tags
    for post in enriched_posts:
        post["tags"] = [tag_mapping.get(tag, tag) for tag in post["tags"]]
        # Remove duplicates while preserving order
        post["tags"] = list(dict.fromkeys(post["tags"]))
    
    # Step 3: Save processed posts
    print("\n--- Step 3: Saving processed posts ---")
    with open(PROCESSED_POSTS_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_posts, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {PROCESSED_POSTS_PATH}")
    
    # Summary
    final_tags = set()
    for post in enriched_posts:
        final_tags.update(post["tags"])
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Total posts processed: {len(enriched_posts)}")
    print(f"Unified tags: {sorted(final_tags)}")
    
    return enriched_posts


if __name__ == "__main__":
    preprocess_posts()
