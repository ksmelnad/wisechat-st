import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)


index_name = "wisechat"
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

st.title("Wise Chat")
st.write("A philosophical journey within you!")
st.write("")
st.write("")

st.text_input("Ask a question:", key="question") 
button_clicked = st.button("Submit", key="button_clicked")
compare_button_clicked = st.button("Compare", key="compare_button_clicked")

question = st.session_state.question

philosophies = {
    "Advaita Vedanta": "advaita-vedanta",
    "Dvaita": "dvaita-vedanta",
    "Vishishtadvaita": "vishishtadvaita-vedanta",
    "Shuddhadvaita": "shuddhadvaita-vedanta",
    "Yoga": "yoga",
    "Zen": "zen"
}

with st.sidebar:
    selected_philosophy_label = st.selectbox("Select Philosophy", 
                 list(philosophies.keys()), key="philosophy")
    st.selectbox("Describe yourself", [
                "Believer", "Spiritualist", "Atheist"], key="user_description")
    st.write("OR")
    st.write("Compare")
    selected_philosophy_label_compare_1 = st.selectbox("Select Philosophy 1", 
                 list(philosophies.keys()), key="philosophy_compare_1")
    selected_philosophy_label_compare_2 = st.selectbox("Select Philosophy 2", 
                 list(philosophies.keys()), key="philosophy_compare_2")



user_description = st.session_state.user_description
selected_philosophy_key = philosophies[selected_philosophy_label]

# st.write("Selected Philosophy:", selected_philosophy_key, type(selected_philosophy_key))





if button_clicked:
    if not question:
        st.warning("Please enter a question.")
        st.stop()

    # st.write("Your question question:", question)
    st.write("Selected Philosophy:", st.session_state.philosophy)

    with st.spinner("Fetching results..."):
        results = vector_store.similarity_search(
            question,
            k=3,
            namespace=selected_philosophy_key
        )
        

    if results:
        with st.expander("Context: Top 3 results:"):
            st.write("")
            for result in results:
                st.write("Result")
                st.write(result)
                st.write("---")
    else:
        st.write("No results found.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers the question 
                strictly only relying on the context provided in generating your response. 
                Never rely on external sources.
                Keep in mind that your are addressing a person who is {user_description} and asking his choice of 
                phisophy is {philosophy}. So your response should be accordingly.
                Response must not be longer than 200 words. If you don't find the answer, say that you don't know.""",
            ),
            ("human", "{question}"),
            ("human", "{context}"),
        ]
    )

    chain = prompt | llm

    with st.spinner("Generating answer..."):
        ai_msg = chain.invoke(
            {
                "context": results,
                "question": question,
                "user_description": user_description,
                "philosophy": selected_philosophy_label
            }
        )
    st.subheader("Answer:")
    st.write(ai_msg.content)    
    

if compare_button_clicked:
    if not question:
        st.warning("Please enter a question.")
        st.stop()
    
    with st.spinner("Fetching results..."):
        results_1 = vector_store.similarity_search(
            question,
            k=3,
            namespace=philosophies[selected_philosophy_label_compare_1]
        )
        results_2 = vector_store.similarity_search(
            question,
            k=3,
            namespace=philosophies[selected_philosophy_label_compare_2]
        )
    if results_1 and results_2:
        with st.expander("Context: Top 3 results:"):
            st.write("")
            for result in results_1:
                st.write("Result 1")
                st.write(result)
                st.write("---")
            for result in results_2:
                st.write("Result 2")
                st.write(result)
                st.write("---")

    else:
        st.write("No results found.")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers the question based on the two contexts provided. You must
                strictly only rely on the contexts provided in generating your response. 
                Never rely on external sources.
                Keep in mind that your are addressing a person who is {user_description} and asking his choice of 
                phisophies {selected_philosophy_label_compare_1} and {selected_philosophy_label_compare_2}. So your response should be accordingly.
                Response must not be longer than 200 words. If you don't find the answer, say that you don't know.""",
            ),
            ("human", "{question}"),
            ("human", "Context 1: {context_1}"),
            ("human", "Context 2: {context_2}"),
        ]
    )

    chain = prompt | llm

    with st.spinner("Generating answer..."):
        ai_msg = chain.invoke(
            {
                "context_1": results_1,
                "context_2": results_2,
                "question": question,
                "user_description": user_description,
                "selected_philosophy_label_compare_1": selected_philosophy_label_compare_1,
                "selected_philosophy_label_compare_2": selected_philosophy_label_compare_2
            }
        )
    st.subheader("Answer:")
    st.write(ai_msg.content)   