"""
Few-Shot Retrieval Module
Loads processed posts and retrieves matching examples for few-shot learning.
"""
import json
import os
import pandas as pd

# Path to processed posts
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_POSTS_PATH = os.path.join(DATA_DIR, "processed_posts.json")


def load_posts() -> pd.DataFrame:
    """Load processed posts into a Pandas DataFrame."""
    if not os.path.exists(PROCESSED_POSTS_PATH):
        raise FileNotFoundError(
            f"Processed posts not found at {PROCESSED_POSTS_PATH}. "
            "Please run 'python preprocess.py' first."
        )
    
    with open(PROCESSED_POSTS_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)
    
    df = pd.DataFrame(posts)
    
    # Add length category
    df["length_category"] = df["line_count"].apply(categorize_length)
    
    return df


def categorize_length(line_count: int) -> str:
    """
    Categorize post length based on line count.
    
    - Short: < 5 lines
    - Medium: 5-10 lines
    - Long: > 10 lines
    """
    if line_count < 5:
        return "Short"
    elif line_count <= 10:
        return "Medium"
    else:
        return "Long"


def get_tags() -> list:
    """
    Get list of unique tags from all processed posts.
    
    Returns:
        list: Sorted list of unique tag names
    """
    df = load_posts()
    
    # Flatten all tags from all posts
    all_tags = []
    for tags in df["tags"]:
        if isinstance(tags, list):
            all_tags.extend(tags)
    
    # Return unique sorted tags
    return sorted(set(all_tags))


def get_languages() -> list:
    """
    Get list of unique languages from processed posts.
    
    Returns:
        list: List of unique languages
    """
    df = load_posts()
    return sorted(df["language"].unique().tolist())


def get_length_categories() -> list:
    """
    Get available length categories.
    
    Returns:
        list: ["Short", "Medium", "Long"]
    """
    return ["Short", "Medium", "Long"]


def get_filtered_posts(
    length: str = None,
    language: str = None,
    tag: str = None,
    max_results: int = 3
) -> list:
    """
    Retrieve posts matching the specified criteria for few-shot examples.
    
    Args:
        length: Length category ("Short", "Medium", "Long")
        language: Language ("English", "Hinglish")
        tag: Topic tag to filter by
        max_results: Maximum number of posts to return
    
    Returns:
        list: List of matching post dictionaries
    """
    df = load_posts()
    
    # Apply filters
    if length:
        df = df[df["length_category"] == length]
    
    if language:
        df = df[df["language"] == language]
    
    if tag:
        # Filter posts that contain the specified tag
        df = df[df["tags"].apply(lambda tags: tag in tags if isinstance(tags, list) else False)]
    
    # If no exact matches, try relaxing filters
    if len(df) == 0:
        print(f"No exact matches for length={length}, language={language}, tag={tag}")
        print("Relaxing filters to find similar posts...")
        
        # Reload and try with just the tag
        df = load_posts()
        if tag:
            df = df[df["tags"].apply(lambda tags: tag in tags if isinstance(tags, list) else False)]
        
        # If still no matches, return any posts
        if len(df) == 0:
            df = load_posts()
    
    # Sort by engagement (higher engagement = better examples)
    df = df.sort_values("engagement", ascending=False)
    
    # Take top results
    results = df.head(max_results).to_dict("records")
    
    return results


def get_post_summary(df: pd.DataFrame = None) -> dict:
    """
    Get a summary of the processed posts dataset.
    
    Returns:
        dict: Summary statistics
    """
    if df is None:
        df = load_posts()
    
    summary = {
        "total_posts": len(df),
        "languages": df["language"].value_counts().to_dict(),
        "length_categories": df["length_category"].value_counts().to_dict(),
        "tags": get_tags(),
        "avg_engagement": df["engagement"].mean(),
        "avg_line_count": df["line_count"].mean()
    }
    
    return summary


if __name__ == "__main__":
    # Test the module
    print("=== Few-Shot Module Test ===\n")
    
    try:
        # Load and display summary
        summary = get_post_summary()
        print(f"Total posts: {summary['total_posts']}")
        print(f"Languages: {summary['languages']}")
        print(f"Length categories: {summary['length_categories']}")
        print(f"Available tags: {summary['tags']}")
        print(f"Avg engagement: {summary['avg_engagement']:.0f}")
        print(f"Avg line count: {summary['avg_line_count']:.1f}")
        
        # Test filtering
        print("\n--- Testing Filters ---")
        
        tags = get_tags()
        if tags:
            test_tag = tags[0]
            posts = get_filtered_posts(length="Medium", language="English", tag=test_tag)
            print(f"\nFiltered posts for tag='{test_tag}', length='Medium', language='English':")
            for i, post in enumerate(posts):
                print(f"  {i+1}. [{post['engagement']} engagements] {post['text'][:80]}...")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python preprocess.py' first to generate processed posts.")
