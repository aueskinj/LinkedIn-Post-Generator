"""
LinkedIn Post Generator - Streamlit Frontend
A tool that generates LinkedIn posts mimicking a specific influencer's writing style.
"""
import os
import streamlit as st
from few_shot import get_tags, get_length_categories, get_languages, get_post_summary
from post_generator import generate_post, generate_post_with_custom_topic

# Paths for auto-preprocessing
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_POSTS_PATH = os.path.join(DATA_DIR, "raw_posts.json")
PROCESSED_POSTS_PATH = os.path.join(DATA_DIR, "processed_posts.json")


def needs_preprocessing():
    """Check if preprocessing is needed (no processed file or raw is newer)."""
    if not os.path.exists(PROCESSED_POSTS_PATH):
        return True
    if not os.path.exists(RAW_POSTS_PATH):
        return False
    # Check if raw posts were modified after processed posts
    raw_mtime = os.path.getmtime(RAW_POSTS_PATH)
    processed_mtime = os.path.getmtime(PROCESSED_POSTS_PATH)
    return raw_mtime > processed_mtime


def run_preprocessing():
    """Run the preprocessing pipeline."""
    from preprocess import preprocess_posts
    with st.spinner("🔄 Preprocessing posts (extracting metadata & unifying tags)..."):
        try:
            preprocess_posts()
            st.success("✅ Preprocessing complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()


# Page configuration
st.set_page_config(
    page_title="LinkedIn Post Generator",
    layout="centered"
)

# Title and description
st.title("LinkedIn Post Generator")
st.markdown("""
Generate engaging LinkedIn posts that match your writing style using AI.
Powered by **Llama 3.2** via Groq with few-shot learning.
""")

st.divider()

# Auto-run preprocessing if needed
if needs_preprocessing():
    st.info("📋 Raw posts detected. Running preprocessing automatically...")
    run_preprocessing()

# Load processed data
try:
    tags = get_tags()
    languages = get_languages()
    length_categories = get_length_categories()
    summary = get_post_summary()
    
    # Show dataset info in expander
    with st.expander("Dataset Info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posts", summary["total_posts"])
        with col2:
            st.metric("Avg Engagement", f"{summary['avg_engagement']:.0f}")
        with col3:
            st.metric("Available Tags", len(summary["tags"]))

except FileNotFoundError:
    st.error("""
    **No posts data found!**
    
    Please add your LinkedIn posts to `data/raw_posts.json` and reload the page.
    """)
    st.stop()

# Input section
st.subheader("Configure Your Post")

# Three columns for dropdowns
col1, col2, col3 = st.columns(3)

with col1:
    selected_topic = st.selectbox(
        "Topic",
        options=tags,
        help="Select a topic based on your past posts"
    )

with col2:
    selected_length = st.selectbox(
        "Length",
        options=length_categories,
        index=1,  # Default to "Medium"
        help="Short (<5 lines), Medium (5-10), Long (>10)"
    )

with col3:
    selected_language = st.selectbox(
        "Language",
        options=languages if languages else ["English"],
        help="Language style for the post"
    )

# Optional: Custom topic input
with st.expander("Use Custom Topic Instead"):
    custom_topic = st.text_input(
        "Enter your custom topic",
        placeholder="e.g., The future of autonomous trading systems"
    )
    use_custom = st.checkbox("Use custom topic instead of dropdown selection")

st.divider()

# Generate button
if st.button("Generate Post", type="primary", use_container_width=True):
    # Determine topic to use
    topic = custom_topic if (use_custom and custom_topic) else selected_topic
    
    if not topic:
        st.warning("Please enter a topic or select one from the dropdown.")
    else:
        with st.spinner("Generating your LinkedIn post..."):
            try:
                # Generate the post
                if use_custom and custom_topic:
                    generated_post = generate_post_with_custom_topic(
                        custom_topic=topic,
                        length=selected_length,
                        language=selected_language,
                        reference_tag=selected_topic  # Use selected tag for style reference
                    )
                else:
                    generated_post = generate_post(
                        topic=topic,
                        length=selected_length,
                        language=selected_language
                    )
                
                # Display the result
                st.success("Post generated successfully!")
                
                st.subheader("Your Generated Post")
                
                # Styled post container
                import html
                escaped_for_html = html.escape(generated_post).replace('\n', '<br>')
                
                st.markdown(f"""
                <div style="
                    background-color: #1a1a2e;
                    color: #e8e8e8;
                    padding: 24px;
                    border-radius: 12px;
                    border-left: 5px solid #0077b5;
                    font-size: 15px;
                    line-height: 1.7;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                ">
                {escaped_for_html}
                </div>
                """, unsafe_allow_html=True)
                
                             
                # Stats
                line_count = len([l for l in generated_post.split('\n') if l.strip()])
                word_count = len(generated_post.split())
                
                st.caption(f"Stats: {line_count} lines | {word_count} words")
                
            except ValueError as e:
                st.error(f"""
                **API Key Error**
                
                {str(e)}
                
                Please add your Groq API key to the `.env` file:
                ```
                GROQ_API_KEY=gsk_your_key_here
                ```
                """)
            except Exception as e:
                st.error(f"Error generating post: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    Built with Streamlit, LangChain & Llama 3.2 by @aueskinj🫡
</div>
""", unsafe_allow_html=True)
