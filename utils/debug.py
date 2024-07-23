def retriever_output_logger(documents):
    print("returning total results count: ", len(documents))
    for doc in documents:
        print(f"""*** {doc.metadata['title']}
            > Prep Time: {doc.metadata['time']}
            > Occasion: {doc.metadata['occasion']}
            > Cuisine: {doc.metadata['cuisine']}
            > Ingredients: {doc.metadata['ingredients']}""")
    return documents
