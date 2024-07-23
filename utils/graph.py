import operator
from pprint import pprint
from typing import Annotated, List, TypedDict
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .db import get_recipes, shortlisted_recipes_to_string
from .graph_chains import (
    get_grader_chain,
    get_recipe_selection_chain,
    get_recipe_url_extractor_chain,
    get_selected_recipe,
)
from .retrievers import get_self_retriever


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # messages: Annotated[Sequence[BaseMessage], add_messages]

    # question: str
    question: Annotated[str, operator.setitem]
    generation: str
    documents: List[str]
    shortlisted_recipes: List[dict]
    selected_recipe: dict
    messages: Annotated[list, add_messages]


def generate_workflow(base_llm):
    def _node_call_retriever(state: AgentState):
        print("---RETRIEVE---")
        question = state["question"]
        vector_db_chain = get_self_retriever(base_llm)
        # Retrieval
        documents = vector_db_chain.invoke(question)
        return {"documents": documents, "question": question}

    def _node_grade_recipes(state: AgentState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        retrieval_grader = get_grader_chain(base_llm)

        # Score each doc
        filtered_docs = []
        for d in documents:
            grader_output = retrieval_grader.invoke({"question": question, "document": d.page_content})
            binary_score = grader_output.binary_score

            if binary_score == "yes":
                score = grader_output.integer_score
                print("---GRADE: DOCUMENT RELEVANT---: ", score)
                d.metadata["score"] = score
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    async def _node_generate_response(state: AgentState, config):
        """
        Determines whether the retrieved recipes are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECKING RECIPE RELEVANCE---")

        question = state["question"]
        documents = state["documents"]

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using the provided context, 
            please answer the user's question in a friendly, conversational tone. 

            Based on the context provided, please select the top 3 receipes that best fits criteria 
            outlined in the question and is most suitable given any other requirements in the question. 
            
            For each option, provide the following information:
            1. A brief description of the recipe
            2. The URL of the recipe
            3. The ratings and number of ratings

            If the context is empty, please be careful to note to the user that there are no recipes matching those specific requirements and do NOT provide any other recipes as suggestions.
            If you don't know the answer based on the context, say you don't know.

            Context:
            {context}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

        chain = base_rag_prompt | base_llm
        full_response = ""
        cl_msg = config["configurable"]["cl_msg"]
        async for chunk in chain.astream(
            {"question": question, "context": documents},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if isinstance(chunk, AIMessageChunk):
                await cl_msg.stream_token(chunk.content)
                full_response += chunk.content
        url_extractor = get_recipe_url_extractor_chain(base_llm)
        url_extractor_results = url_extractor.invoke({"context": full_response})

        shortlisted_recipes = None
        if isinstance(url_extractor_results.urls, list) and len(url_extractor_results.urls):
            shortlisted_recipes = get_recipes(url_extractor_results.urls)
        return {
            "documents": documents,
            "question": question,
            "shortlisted_recipes": shortlisted_recipes,
            "messages": [full_response],
        }

    async def _node_shortlist_qa(state: AgentState, config):
        print("--- Q&A with SHORTLISTED RECIPES ---")

        question = state["question"]
        shortlisted_recipes = state["shortlisted_recipes"]
        messages = state["messages"]

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using only the provided context, 
            please answer the user's question in a friendly, conversational tone.
            If you don't know the answer based on the context, say you don't know.

            Context:
            {context}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

        chain = base_rag_prompt | base_llm
        full_response = ""
        cl_msg = config["configurable"]["cl_msg"]
        async for chunk in chain.astream(
            {"question": question, "context": shortlisted_recipes_to_string(shortlisted_recipes)},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if isinstance(chunk, AIMessageChunk):
                await cl_msg.stream_token(chunk.content)
                full_response += chunk.content

        selected_recipe = get_selected_recipe(base_llm, question, shortlisted_recipes, messages)

        return {"messages": [full_response], "selected_recipe": selected_recipe}

    async def _node_single_recipe_qa(state: AgentState, config):
        print("--- Q&A with SINGLE RECIPE ---")

        question = state["question"]
        shortlisted_recipes = state["shortlisted_recipes"]
        selected_recipe = state.get("selected_recipe")
        messages = state["messages"]
        if not selected_recipe:
            selected_recipe = get_selected_recipe(base_llm, question, shortlisted_recipes, messages)
        print("selected_recipe", selected_recipe)

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using only the provided context, 
            please answer the user's question in a friendly, conversational tone.
            If you don't know the answer based on the context, say you don't know.

            Context:
            {context}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

        chain = base_rag_prompt | base_llm
        full_response = ""
        cl_msg = config["configurable"]["cl_msg"]
        async for chunk in chain.astream(
            {"question": question, "context": selected_recipe["text"]},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if isinstance(chunk, AIMessageChunk):
                await cl_msg.stream_token(chunk.content)
                full_response += chunk.content

        return {"messages": [full_response]}

    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("retrieve", _node_call_retriever)  # retrieve
    workflow.add_node("grade_recipes", _node_grade_recipes)  # grade documents
    workflow.add_node("generate", _node_generate_response)  # generatae
    workflow.add_node("shortlist_qa", _node_shortlist_qa)  # answer questions about shortlisted recipes
    workflow.add_node("single_qa", _node_single_recipe_qa)  # answer questions about shortlisted recipes

    # Define the edges
    # workflow.add_edge(START, "retrieve")

    def _edge_route_question(state: AgentState):
        print("=======EDGE: START =====")
        question = state["question"]
        messages = state["messages"]
        last_message = messages[-1] if messages else ""
        shortlisted_recipes = state.get("shortlisted_recipes")
        if not shortlisted_recipes or len(shortlisted_recipes) == 0:
            print("going to retrieve since no shortlisted_recipes")
            return "retrieve"
        recipe_selection_chain = get_recipe_selection_chain(base_llm)
        recipe_selection_response = recipe_selection_chain.invoke(
            {
                "question": question,
                "context": shortlisted_recipes_to_string(shortlisted_recipes),
                "last_message": last_message,
            }
        )
        print("latest message", last_message)

        pprint(recipe_selection_response)
        if recipe_selection_response.asking_for_recipe_suggestions == "yes":
            return "retrieve"
        if (
            recipe_selection_response.referring_to_shortlisted_recipes == "yes"
            or recipe_selection_response.show_specific_recipe == "yes"
        ):
            return "shortlist_qa"
        if (
            recipe_selection_response.referring_to_specific_recipe == "yes"
            and recipe_selection_response.specific_recipe_url
        ):
            return "single_qa"

        print("defaulting to shortlist_qa")
        return "shortlist_qa"

    workflow.add_conditional_edges(
        START,
        _edge_route_question,
        {
            "shortlist_qa": "shortlist_qa",
            "single_qa": "single_qa",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_recipes")
    workflow.add_edge("grade_recipes", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("shortlist_qa", END)
    workflow.add_edge("single_qa", END)

    memory = AsyncSqliteSaver.from_conn_string(":memory:")

    app = workflow.compile(checkpointer=memory)

    return app
