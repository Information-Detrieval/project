# import os
# import pinecone

# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# from langchain import HuggingFaceHub, LLMChain
# from langchain.embeddings import HuggingFaceEmbeddings

# model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# embed = HuggingFaceEmbeddings(
#     model_name=model_name,
# )

# import os
# # os.environ['PINECONE_API_KEY'] = 'b666ad2a-1d89-4252-981b-c6a222a980db'
# os.environ['PINECONE_API_KEY'] = '8a73267f-d64d-4d53-a5ae-0a241afd5517'

# #396bf1fb-9ddf-4e69-a697-6c145324bea8
# # os.environ['PINECONE_ENVIRONMENT'] = 'asia-southeast1-gcp-free'

# import os
# # os.environ['PINECONE_API_KEY'] = 'b666ad2a-1d89-4252-981b-c6a222a980db'
# # os.environ['PINECONE_API_KEY'] = '396bf1fb-9ddf-4e69-a697-6c145324bea8'

# #396bf1fb-9ddf-4e69-a697-6c145324bea8
# # os.environ['PINECONE_ENVIRONMENT'] = 'asia-southeast1-gcp-free'


# from langchain.vectorstores import Pinecone

# text_field = "text"

# # switch back to normal index for langchain
# index_name="detrieval"
# index = pinecone.Index(index_name)

# vectorstore = Pinecone(
#     index, embed.embed_query, text_field
# )



# query = "Who is Dhruv Kumar ?"

# relevant_documents = vectorstore.similarity_search(
#     query,  # our search query
#     k=3  # return 3 most relevant docs
# )

# print(relevant_documents[0])

import pickle
with open ("website_data/pkl/documents.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)