import os
from typing import List
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import (
    MyScale,
    MyScaleSettings,
)
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant.vectorstores import Qdrant

from .metadata import CUISINES, DIETS, EQUIPMENT, KEY_INGREDIENTS, OCCASIONS

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#QDRANT_CLOUD_KEY = os.environ.get("QDRANT_CLOUD_KEY")
#QDRANT_CLOUD_URL = "https://30591e3d-7092-41c4-95e1-4d3c7ef6e894.us-east4-0.gcp.cloud.qdrant.io"


# Define embedding model
base_embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)


def get_ensemble_retriever():
    # Use a Qdrant VectorStore to embed and store our data
    qdrant_descriptions = Qdrant.from_existing_collection(
        embedding=base_embeddings_model,
        # 3 vector indices - recipe_descriptions, recipe_nutrition, recipe_ingredients
        collection_name="recipe_descriptions",
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_CLOUD_KEY,
    )

    qdrant_nutrition = Qdrant.from_existing_collection(
        embedding=base_embeddings_model,
        collection_name="recipe_nutrition",
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_CLOUD_KEY,
    )

    qdrant_ingredients = Qdrant.from_existing_collection(
        embedding=base_embeddings_model,
        collection_name="recipe_ingredients",
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_CLOUD_KEY,
    )

    # Convert retrieved documents to JSON-serializable format
    descriptions_retriever = qdrant_descriptions.as_retriever(search_kwargs={"k": 20})
    nutrition_retriever = qdrant_nutrition.as_retriever(search_kwargs={"k": 20})
    ingredients_retriever = qdrant_ingredients.as_retriever(search_kwargs={"k": 20})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            descriptions_retriever,
            nutrition_retriever,
            ingredients_retriever,
        ],
        weights=[
            0.5,
            0.25,
            0.25,
        ],
    )

    return ensemble_retriever


def _list_to_string(l: list) -> str:
    return ", ".join([f"`{item}`" for item in l])


class ModifiedSelfQueryRetriever(SelfQueryRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        structured_query = self.query_constructor.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )
        # if self.verbose:
        #     logger.info(f"Generated Query: {structured_query}")

        new_query, search_kwargs = self._prepare_query(query, structured_query)

        print("search_kwargs", search_kwargs)
        self.search_kwargs = search_kwargs

        docs = self._get_docs_with_query(new_query, search_kwargs)
        return docs


def get_self_retriever(llm_model):
    metadata_field_info = [
        AttributeInfo(
            name="cuisine",
            description="The national / ethnic cuisine categories of the recipe."
            f"It should be one of {_list_to_string(CUISINES)}. "
            "It only supports contain comparisons. "
            f"Here are some examples: contain (cuisine, '{CUISINES[0]}')",
            type="list[string]",
        ),
        AttributeInfo(
            name="diet",
            description="The diets / dietary restrictions satisfied by this recipe."
            f"It should be one of {_list_to_string(DIETS)}. "
            "It only supports contain comparisons. "
            f"Here are some examples: contain (diet, '{DIETS[0]}')",
            type="list[string]",
        ),
        AttributeInfo(
            name="equipment",
            description="The equipment required by this recipe."
            f"It should be one of {_list_to_string(EQUIPMENT)}. "
            "It only supports contain comparisons. "
            f"Here are some examples: contain (equipment, '{EQUIPMENT[0]}')",
            type="list[string]",
        ),
        AttributeInfo(
            name="occasion",
            description="The occasions, holidays, celebrations that are well suited for this recipe."
            f"It should be one of {_list_to_string(OCCASIONS)}. "
            "It only supports contain comparisons. "
            f"Here are some examples: contain (occasion, '{OCCASIONS[0]}')",
            type="list[string]",
        ),
        # AttributeInfo(
        #     name="ingredients",
        #     description="The ingredients used to make this recipe."
        #     f"It should be one of {_list_to_string(KEY_INGREDIENTS)}"
        #     "It only supports contain comparisons. "
        #     f"Here are some examples: contain (ingredients, '{KEY_INGREDIENTS[0]}')",
        #     type="list[string]",
        # ),
        AttributeInfo(
            name="time",
            description="The estimated time in minutes required to cook and prepare the recipe",
            type="integer",
        ),
    ]

    config = MyScaleSettings(
        host=os.environ["MYSCALE_HOST"],
        port=443,
        username=os.environ["MYSCALE_USERNAME"],
        password=os.environ["MYSCALE_PASSWORD"],
    )
    vectorstore = MyScale(base_embeddings_model, config)

    retriever = ModifiedSelfQueryRetriever.from_llm(
        llm_model,
        vectorstore,
        "Brief summary and key attributes of a recipe, including ingredients, cooking time, occasion, cuisine and diet",
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 10},
    )
    return retriever
