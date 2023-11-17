import os
import streamlit
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
palm.configure(api_key=api_key)


def main():
    streamlit.header("ASK YOUR PDF")

    pdf = streamlit.file_uploader("Upload a PDF File", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        streamlit.write(store_name)

        embeddings = GooglePalmEmbeddings(model_name="models/embedding-gecko-001")
        try:
            vector_store = FAISS.load_local(store_name, embeddings)
            print("Loaded vector from local")
        except RuntimeError:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(store_name)

        query = streamlit.text_input("Ask questions about your PDF file:")
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = GooglePalm(model_name="models/text-bison-001")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            streamlit.write(response)


if __name__ == '__main__':
    main()