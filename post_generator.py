"""
Post Generator Module
Constructs prompts with few-shot examples and generates LinkedIn posts.
"""
from langchain_core.prompts import PromptTemplate
from llm_helper import get_llm
from few_shot import get_filtered_posts


def create_post_prompt():
    """Create the main prompt template for post generation."""
    template = """You are a LinkedIn content creator who writes engaging, insightful posts about data science, quantitative finance, and technology.

Your task is to write a NEW LinkedIn post about the topic: {topic}

CONSTRAINTS:
- Language: {language}
- Length: {length} ({length_description})
- Write in a professional yet conversational tone
- Include thought-provoking insights or contrarian takes
- End with a question or call-to-action to encourage engagement

WRITING STYLE EXAMPLES:
Study these example posts carefully and mimic their writing style, structure, and tone:

{examples}

---

Now write a NEW, ORIGINAL LinkedIn post about "{topic}" following the same style.
Do not copy the examples - create fresh content that captures the same voice and approach.

YOUR POST:"""
    
    return PromptTemplate(
        input_variables=["topic", "language", "length", "length_description", "examples"],
        template=template
    )


def format_examples(posts: list) -> str:
    """Format example posts for inclusion in the prompt."""
    if not posts:
        return "(No matching examples found - write in a professional, insightful style)"
    
    formatted = []
    for i, post in enumerate(posts, 1):
        formatted.append(f"--- Example {i} ---\n{post['text']}\n")
    
    return "\n".join(formatted)


def get_length_description(length: str) -> str:
    """Get a description of the target length."""
    descriptions = {
        "Short": "4 lines or less - punchy and concise",
        "Medium": "5-10 lines - balanced depth and brevity",
        "Long": "11+ lines - comprehensive exploration of the topic"
    }
    return descriptions.get(length, "Medium length")


def generate_post(
    topic: str,
    length: str = "Medium",
    language: str = "English"
) -> str:
    """
    Generate a LinkedIn post matching the specified criteria.
    
    Args:
        topic: The topic/tag to write about
        length: Post length ("Short", "Medium", "Long")
        language: Language ("English", "Hinglish")
    
    Returns:
        str: Generated LinkedIn post
    """
    # Get few-shot examples matching the criteria
    example_posts = get_filtered_posts(
        length=length,
        language=language,
        tag=topic,
        max_results=3
    )
    
    # Format examples for the prompt
    examples_text = format_examples(example_posts)
    length_description = get_length_description(length)
    
    # Create and invoke the chain
    prompt = create_post_prompt()
    llm = get_llm()
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": topic,
        "language": language,
        "length": length,
        "length_description": length_description,
        "examples": examples_text
    })
    
    return result.content


def generate_post_with_custom_topic(
    custom_topic: str,
    length: str = "Medium",
    language: str = "English",
    reference_tag: str = None
) -> str:
    """
    Generate a post with a custom topic (not from predefined tags).
    
    Args:
        custom_topic: Custom topic to write about
        length: Post length
        language: Language
        reference_tag: Optional tag to use for fetching style examples
    
    Returns:
        str: Generated LinkedIn post
    """
    # Get examples based on reference tag or general high-engagement posts
    if reference_tag:
        example_posts = get_filtered_posts(
            length=length,
            language=language,
            tag=reference_tag,
            max_results=3
        )
    else:
        # Get top posts by engagement regardless of tag
        example_posts = get_filtered_posts(
            length=length,
            language=language,
            max_results=3
        )
    
    examples_text = format_examples(example_posts)
    length_description = get_length_description(length)
    
    prompt = create_post_prompt()
    llm = get_llm()
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": custom_topic,
        "language": language,
        "length": length,
        "length_description": length_description,
        "examples": examples_text
    })
    
    return result.content


if __name__ == "__main__":
    # Test the generator
    print("=== Post Generator Test ===\n")
    
    try:
        from few_shot import get_tags
        
        tags = get_tags()
        if tags:
            test_tag = tags[0]
            print(f"Generating a Medium, English post about: {test_tag}\n")
            
            post = generate_post(
                topic=test_tag,
                length="Medium",
                language="English"
            )
            
            print("--- Generated Post ---")
            print(post)
        else:
            print("No tags available. Please run preprocessing first.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your GROQ_API_KEY in .env")
        print("2. Run 'python preprocess.py' first")
