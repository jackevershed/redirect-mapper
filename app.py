"""
AI Redirect Mapper - Streamlit App
Upload and run: streamlit run app.py
"""

import streamlit as st
from google import genai
import json
import csv
import time
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO
import tempfile

# Page configuration
st.set_page_config(
    page_title="Jaywing AI Redirect Mapper",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to ensure scrolling works
st.markdown("""
    <style>
    .main {
        overflow-y: auto !important;
    }
    section[data-testid="stSidebar"] {
        overflow-y: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matches' not in st.session_state:
    st.session_state.matches = []
if 'crawl_results_old' not in st.session_state:
    st.session_state.crawl_results_old = []
if 'crawl_results_new' not in st.session_state:
    st.session_state.crawl_results_new = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'auth_required_urls' not in st.session_state:
    st.session_state.auth_required_urls = []
if 'credentials' not in st.session_state:
    st.session_state.credentials = {}

# Checkpoint directory
CHECKPOINT_DIR = tempfile.gettempdir() + '/redirect_mapper_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Helper functions
def save_checkpoint(data, filename):
    """Save checkpoint to temp directory"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint(filename):
    """Load checkpoint from temp directory"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def parse_urls(content):
    """Parse URLs from uploaded file"""
    lines = content.decode('utf-8').strip().split('\n')
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

def fetch_page_content(url, retry_count=0, max_retries=3, credentials=None):
    """Fetch page content with retry logic and authentication support"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    ]
    
    headers = {
        'User-Agent': user_agents[retry_count % len(user_agents)],
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Prepare authentication if credentials provided
    auth = None
    if credentials:
        from requests.auth import HTTPBasicAuth
        auth = HTTPBasicAuth(credentials['username'], credentials['password'])
    
    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True, auth=auth)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        h1 = soup.find('h1')
        h1_text = h1.get_text().strip() if h1 else ""
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_text = meta_desc.get('content', '').strip() if meta_desc else ""
        
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')[:3]]
        content_preview = ' '.join(paragraphs)[:500]
        
        return {
            'url': url,
            'title': title_text,
            'heading': h1_text,
            'description': meta_text,
            'content': content_preview,
            'status': 'success',
            'error': None
        }
        
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            time.sleep(2 ** retry_count)
            return fetch_page_content(url, retry_count + 1, max_retries, credentials)
        return {'url': url, 'title': '', 'heading': '', 'description': '', 'content': '', 
                'status': 'failed', 'error': 'Timeout'}
        
    except requests.exceptions.HTTPError as e:
        # Check for authentication errors
        if e.response.status_code in [401, 403]:
            return {'url': url, 'title': '', 'heading': '', 'description': '', 'content': '', 
                    'status': 'auth_required', 'error': f'Authentication required (HTTP {e.response.status_code})'}
        
        if retry_count < max_retries and e.response.status_code in [429, 500, 502, 503, 504]:
            time.sleep(3 ** retry_count)
            return fetch_page_content(url, retry_count + 1, max_retries, credentials)
        return {'url': url, 'title': '', 'heading': '', 'description': '', 'content': '', 
                'status': 'failed', 'error': f'HTTP {e.response.status_code}'}
        
    except Exception as e:
        return {'url': url, 'title': '', 'heading': '', 'description': '', 'content': '', 
                'status': 'failed', 'error': str(e)[:100]}

def crawl_urls(url_list, progress_bar, status_text, credentials=None):
    """Crawl URLs with progress updates and authentication support"""
    results = []
    auth_required = []
    total = len(url_list)
    
    for i, url in enumerate(url_list):
        status_text.text(f"Crawling {i + 1}/{total}: {url[:50]}...")
        progress_bar.progress((i + 1) / total)
        
        content = fetch_page_content(url, credentials=credentials)
        results.append(content)
        
        # Track URLs that require authentication
        if content['status'] == 'auth_required':
            auth_required.append(url)
        
        time.sleep(1.0)
    
    return results, auth_required

def match_urls(client, model_name, old_data, new_data, crawled, progress_bar, status_text):
    """Match URLs using Gemini API"""
    all_matches = []
    
    if not crawled:
        old_data = [{'url': url} for url in old_data]
        new_data = [{'url': url} for url in new_data]
    
    if crawled:
        old_data = [item for item in old_data if item['status'] == 'success']
    
    batch_size = 100 if 'flash' in model_name else 50
    total_batches = (len(old_data) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(old_data))
        old_batch = old_data[start_idx:end_idx]
        
        status_text.text(f"Processing batch {batch_num + 1}/{total_batches} ({len(old_batch)} URLs)...")
        progress_bar.progress((batch_num + 1) / total_batches)
        
        if crawled:
            old_urls_text = '\n\n'.join([
                f"URL: {item['url']}\nTitle: {item['title']}\nHeading: {item['heading']}\nDescription: {item['description']}\nContent: {item['content']}"
                for item in old_batch
            ])
            new_urls_text = '\n\n'.join([
                f"URL: {item['url']}\nTitle: {item['title']}\nHeading: {item['heading']}\nDescription: {item['description']}\nContent: {item['content']}"
                for item in new_data if item['status'] == 'success'
            ])
            
            prompt = f"""You are a URL redirect mapping expert. Match old URLs to new URLs based on semantic similarity, content analysis, and URL structure.

Old URLs with content:
{old_urls_text}

New URLs with content:
{new_urls_text}

Return ONLY a JSON array with this exact structure (no markdown, no preamble):
[
  {{
    "oldUrl": "old-url-here",
    "newUrl": "new-url-here",
    "confidence": 0.95,
    "reason": "brief reason for match based on content similarity"
  }}
]

Match as many old URLs as possible. Use confidence scores: 1.0 = perfect match, 0.7-0.9 = good match, 0.5-0.6 = uncertain match."""
        else:
            old_urls_text = '\n'.join([item['url'] for item in old_batch])
            new_urls_text = '\n'.join([item['url'] for item in new_data])
            
            prompt = f"""You are a URL redirect mapping expert. Match old URLs to new URLs based on semantic similarity, content keywords, and URL structure.

Old URLs:
{old_urls_text}

New URLs:
{new_urls_text}

Return ONLY a JSON array with this exact structure (no markdown, no preamble):
[
  {{
    "oldUrl": "old-url-here",
    "newUrl": "new-url-here",
    "confidence": 0.95,
    "reason": "brief reason for match"
  }}
]

Match as many old URLs as possible. Use confidence scores: 1.0 = perfect match, 0.7-0.9 = good match, 0.5-0.6 = uncertain match."""
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            response_text = response.text
            
            clean_text = response_text.replace('```json', '').replace('```', '').strip()
            batch_matches = json.loads(clean_text)
            all_matches.extend(batch_matches)
            
        except Exception as e:
            st.error(f"Error in batch {batch_num + 1}: {str(e)}")
            continue
        
        time.sleep(1.5)
    
    return all_matches

def export_csv(matches):
    """Export matches to CSV"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Old URL', 'New URL', 'Confidence', 'Reason'])
    for m in matches:
        writer.writerow([m['oldUrl'], m['newUrl'], m['confidence'], m['reason']])
    return output.getvalue()

# Main UI
st.title("AI Redirect Mapper")
st.markdown("Upload your old and new URL lists, and let AI match them using Gemini 3")

# Sidebar configuration
with st.sidebar:
    st.image("https://d3q27bh1u24u2o.cloudfront.net/news/jay_1.jpg", width=120)
    st.header("Configuration")
    
    api_key = st.text_input("Google AI Studio API Key", type="password", 
                            help="Get one at: https://aistudio.google.com/apikey")
    
    st.divider()
    
    model_choice = st.radio(
        "Select Model",
        options=[
            "gemini-3-flash-preview",
            "gemini-3-pro-preview"
        ],
        format_func=lambda x: "Gemini 3 Flash (Recommended - Fast & Intelligent)" if "flash" in x 
                              else "Gemini 3 Pro (Best Quality)",
        help="Flash: Best balance. Pro: Highest quality for complex matching."
    )
    
    crawl_enabled = st.checkbox(
        "Crawl page content",
        help="Fetch actual page content for more accurate matching. Slower but much better results."
    )
    
    # Authentication section
    with st.expander("Authentication (Optional)", expanded=False):
        st.caption("Provide credentials if your URLs require authentication")
        auth_username = st.text_input("Username", key="auth_user")
        auth_password = st.text_input("Password", type="password", key="auth_pass")
        
        if auth_username and auth_password:
            st.session_state.credentials = {
                'username': auth_username,
                'password': auth_password
            }
            st.success("Credentials configured")
        else:
            st.session_state.credentials = {}
    
    st.divider()
    
    if st.button("Clear Checkpoints", type="secondary"):
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            st.success("Checkpoints cleared!")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Old URLs")
    old_file = st.file_uploader("Upload old URLs (CSV or TXT)", type=["csv", "txt"], key="old")
    if old_file:
        old_urls = parse_urls(old_file.read())
        st.success(f"Loaded {len(old_urls)} old URLs")
        with st.expander("Preview old URLs"):
            st.code('\n'.join(old_urls[:10]))
            if len(old_urls) > 10:
                st.caption(f"... and {len(old_urls) - 10} more")

with col2:
    st.subheader("New URLs")
    new_file = st.file_uploader("Upload new URLs (CSV or TXT)", type=["csv", "txt"], key="new")
    if new_file:
        new_urls = parse_urls(new_file.read())
        st.success(f"Loaded {len(new_urls)} new URLs")
        with st.expander("Preview new URLs"):
            st.code('\n'.join(new_urls[:10]))
            if len(new_urls) > 10:
                st.caption(f"... and {len(new_urls) - 10} more")

st.divider()

# Start matching button
start_button_disabled = st.session_state.processing or not (old_file and new_file and api_key)

# Show completion message if processing just finished
if st.session_state.processing_complete and len(st.session_state.matches) > 0:
    st.success("✓ Processing complete! Scroll down to see results and download CSV.")

if st.button("Start Matching", type="primary", disabled=start_button_disabled):
    if not api_key:
        st.error("Please enter your Google AI Studio API key in the sidebar")
    elif not old_file or not new_file:
        st.error("Please upload both old and new URL files")
    else:
        st.session_state.processing = True
        st.session_state.processing_complete = False
        st.session_state.matches = []  # Clear previous results
        st.session_state.auth_required_urls = []  # Clear previous auth errors
        
        try:
            # Initialize client
            client = genai.Client(api_key=api_key)
            
            # Crawling phase
            if crawl_enabled:
                st.header("Crawling Pages")
                
                # Get credentials if provided
                creds = st.session_state.credentials if st.session_state.credentials else None
                
                crawl_container = st.container()
                with crawl_container:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Old URLs")
                        progress_old = st.progress(0)
                        status_old = st.empty()
                        crawl_old_results, auth_old = crawl_urls(old_urls, progress_old, status_old, creds)
                        st.session_state.crawl_results_old = crawl_old_results
                        status_old.empty()  # Clear status after completion
                        
                        successful_old = sum(1 for r in st.session_state.crawl_results_old if r['status'] == 'success')
                        auth_required_old = sum(1 for r in st.session_state.crawl_results_old if r['status'] == 'auth_required')
                        
                        st.success(f"Crawled {successful_old}/{len(old_urls)} old URLs successfully")
                        if auth_required_old > 0:
                            st.warning(f"{auth_required_old} URLs require authentication")
                            st.session_state.auth_required_urls.extend(auth_old)
                    
                    with col2:
                        st.subheader("New URLs")
                        progress_new = st.progress(0)
                        status_new = st.empty()
                        crawl_new_results, auth_new = crawl_urls(new_urls, progress_new, status_new, creds)
                        st.session_state.crawl_results_new = crawl_new_results
                        status_new.empty()  # Clear status after completion
                        
                        successful_new = sum(1 for r in st.session_state.crawl_results_new if r['status'] == 'success')
                        auth_required_new = sum(1 for r in st.session_state.crawl_results_new if r['status'] == 'auth_required')
                        
                        st.success(f"Crawled {successful_new}/{len(new_urls)} new URLs successfully")
                        if auth_required_new > 0:
                            st.warning(f"{auth_required_new} URLs require authentication")
                            st.session_state.auth_required_urls.extend(auth_new)
                
                # Show authentication warning if needed
                if len(st.session_state.auth_required_urls) > 0:
                    st.error(f"⚠️ {len(st.session_state.auth_required_urls)} URLs require authentication and could not be crawled.")
                    with st.expander("View URLs requiring authentication"):
                        for url in st.session_state.auth_required_urls:
                            st.code(url)
                    st.info("💡 To crawl these URLs: Add credentials in the sidebar under 'Authentication (Optional)' and re-run the matching process.")
                
                old_data = st.session_state.crawl_results_old
                new_data = st.session_state.crawl_results_new
            else:
                old_data = old_urls
                new_data = new_urls
            
            # Matching phase
            match_container = st.container()
            with match_container:
                st.header("Matching URLs with AI")
                progress_match = st.progress(0)
                status_match = st.empty()
                
                st.session_state.matches = match_urls(
                    client, 
                    model_choice, 
                    old_data, 
                    new_data, 
                    crawl_enabled,
                    progress_match,
                    status_match
                )
                
                status_match.empty()  # Clear status after completion
                st.success(f"Matching complete! Found {len(st.session_state.matches)} matches")
            
            st.session_state.processing_complete = True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.processing_complete = False
        finally:
            st.session_state.processing = False
            # Force rerun to show results
            st.rerun()

# Display results
if len(st.session_state.matches) > 0:
    st.divider()
    
    # Add a marker to ensure we can scroll to results
    results_anchor = st.container()
    
    with results_anchor:
        st.header("✓ Results")
        
        # Show authentication warning if applicable
        if len(st.session_state.auth_required_urls) > 0:
            st.warning(f"⚠️ Note: {len(st.session_state.auth_required_urls)} URLs were excluded from matching due to authentication requirements.")
        
        # Summary stats
        high_conf = sum(1 for m in st.session_state.matches if m['confidence'] >= 0.8)
        medium_conf = sum(1 for m in st.session_state.matches if 0.6 <= m['confidence'] < 0.8)
        low_conf = sum(1 for m in st.session_state.matches if m['confidence'] < 0.6)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Matches", len(st.session_state.matches))
        col2.metric("High Confidence (80%+)", high_conf)
        col3.metric("Medium Confidence (60-80%)", medium_conf)
        col4.metric("Low Confidence (<60%)", low_conf)
        
        # Download button at top
        st.subheader("Download Results")
        csv_data = export_csv(st.session_state.matches)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"redirects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
        # Preview matches
        st.subheader("Preview Matches")
        preview_count = min(10, len(st.session_state.matches))
        
        for i, match in enumerate(st.session_state.matches[:preview_count], 1):
            with st.expander(f"Match {i} - Confidence: {int(match['confidence']*100)}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text("Old URL:")
                    st.code(match['oldUrl'])
                with col2:
                    st.text("New URL:")
                    st.code(match['newUrl'])
                st.caption(f"Reason: {match['reason']}")
        
        if len(st.session_state.matches) > preview_count:
            st.caption(f"... and {len(st.session_state.matches) - preview_count} more matches")
        
        # Download button at bottom too
        st.divider()
        csv_data_bottom = export_csv(st.session_state.matches)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data_bottom,
            file_name=f"redirects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="download_bottom"
        )

# Footer
st.divider()
st.caption("Powered by Gemini 3 | Jack Evershed")
