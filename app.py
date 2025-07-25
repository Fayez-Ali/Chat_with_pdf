import streamlit as st
from rag_utils import get_vector_db
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, PDF_PATH

print("\nStarting application...")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    print("Chat history initialized")

# Initialize vector DB and LLM
print("\nInitializing components...")
print("1. Loading vector database")
vector_db = get_vector_db()

print("2. Connecting to Gemini")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
print("✓ All components ready\n")

# Streamlit UI
st.title("📄 Chat with PDF")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the document"):
    print(f"\nUser query: {prompt}")
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        print("- Searching for relevant content...")
        docs = vector_db.similarity_search(prompt, k=3)
        print(f"  Found {len(docs)} relevant chunks")
        
        context = "\n\n".join(d.page_content for d in docs)
        
        print("- Generating response...")
        response = llm.invoke(
            f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        )
        print("✓ Response generated")
        
        # Display messages
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})

print("\nApplication running - waiting for queries...")