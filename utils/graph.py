import json
import operator
from pprint import pprint
from typing import Annotated, List, TypedDict
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessageChunk, FunctionMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from utils.tools import send_text_tool
from .db import get_recipes, shortlisted_recipes_to_string
from .graph_chains import (
    get_grader_chain,
    get_question_type_chain,
    get_recipe_url_extractor_chain,
    get_selected_recipe,
)
from .retrievers import get_self_retriever


class AgentState(TypedDict):
    question: Annotated[str, operator.setitem]
    question_type: str
    generation: str
    documents: List[str]
    shortlisted_recipes: List[dict]
    selected_recipe: dict
    messages: Annotated[list, add_messages]


def generate_workflow(base_llm, power_llm):
    def _node_question_triage(state: AgentState):
        print("---TRIAGE---")
        question = state["question"]
        messages = state["messages"]
        last_message = messages[-1] if messages else ""
        shortlisted_recipes = state.get("shortlisted_recipes")
        question_type_chain = get_question_type_chain(base_llm)
        question_type_response = question_type_chain.invoke(
            {
                "question": question,
                "context": shortlisted_recipes_to_string(shortlisted_recipes),
                "last_message": last_message,
            }
        )
        question_type_response_data = sorted(
            [
                (question_type_response.send_text, "send_sms"),
                (question_type_response.asking_for_recipe_suggestions, "asking_for_recipe_suggestions"),
                (question_type_response.referring_to_shortlisted_recipes, "referring_to_shortlisted_recipes"),
                (question_type_response.show_specific_recipe, "show_specific_recipe"),
                (question_type_response.referring_to_specific_recipe, "referring_to_specific_recipe"),
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        pprint(question_type_response_data)
        question_type = question_type_response_data[0][1]
        selected_recipe = None
        if shortlisted_recipes and question_type_response.specific_recipe_url:
            selected_recipe = next(
                (r for r in shortlisted_recipes if r["url"] == question_type_response.specific_recipe_url)
            )
            print("set selected recipe", question_type_response.specific_recipe_url)
        return {"question_type": question_type, "selected_recipe": selected_recipe}

    async def _node_call_retriever(state: AgentState, config):
        print("---RETRIEVE---")
        cl_msg = config["configurable"]["cl_msg"]
        await cl_msg.stream_token("Searching for recipes matching your criteria ... \n\n")
        question = state["question"]
        vector_db_chain = get_self_retriever(base_llm)
        # Retrieval
        documents = vector_db_chain.invoke(question, return_only_outputs=False)
        print("WOW: ", vector_db_chain.search_kwargs)
        return {"documents": documents, "question": question}

    async def _node_grade_recipes(state: AgentState, config):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        cl_msg = config["configurable"]["cl_msg"]
        question = state["question"]
        documents = state["documents"]
        await cl_msg.stream_token(
            f"Evaluating the relevance of {len(documents)} retrieved recipes based on your criteria ... \n\n"
        )

        retrieval_grader = get_grader_chain(base_llm)

        # Score each doc
        filtered_docs = []
        for d in documents:
            grader_output = retrieval_grader.invoke({"question": question, "document": d.page_content})
            binary_score = grader_output.binary_score
            score = grader_output.integer_score

            if binary_score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---: ", score, d.metadata["url"])
                d.metadata["score"] = score
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---", score, d.metadata["url"])
                continue
        num_eliminated_docs = len(documents) - len(filtered_docs)
        if num_eliminated_docs > 0:
            await cl_msg.stream_token(
                f"Eliminated {num_eliminated_docs} recipes that were not relevant based on your criteria ... \n\n"
            )
        return {"documents": filtered_docs, "question": question}

    async def _node_generate_response(state: AgentState, config):
        """
        Determines whether the retrieved recipes are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("--- GENERATING SHORTLIST ---")

        question = state["question"]
        documents = state["documents"]

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using the provided context, 
            please answer the user's question in a friendly, conversational tone. 

            Based on the context provided, please select the top 3 receipes that best fits criteria 
            outlined in the question. It doesn't need to be a perfect match but just get the most suitable.
            
            For each option, provide the following information:
            1. A brief description of the recipe
            2. The URL of the recipe
            3. The ratings and number of ratings
            Only if question includes a criteria for recipes that are good for a specific occassion, please also provide the occassion(s) of the recipe,
            Only if question includes a criteria a type of cuisine, please also provide the cuisines associated with the recipe.
            Only if question includes a criteria a type of diet, please also provide the diet(s) associated with the recipe.

            If the context is empty, please be careful to note to the user that there are no recipes matching those specific requirements and do NOT provide any other recipes as suggestions.

            Context:
            {context}
            
            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

        chain = base_rag_prompt | power_llm
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
        last_message = messages[-1] if messages else ""

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using only the provided context, 
            please answer the user's question in a friendly, conversational tone.
            If you don't know the answer based on the context, say you don't know.

            Context:
            {context}

            Last message provided to the user:
            {last_message}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)
        chain = base_rag_prompt | power_llm

        full_response = ""
        cl_msg = config["configurable"]["cl_msg"]
        async for chunk in chain.astream(
            {
                "question": question,
                "context": shortlisted_recipes_to_string(shortlisted_recipes),
                "last_message": last_message,
            },
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
        selected_recipe = state.get("selected_recipe")
        messages = state["messages"]
        last_message = messages[-1] if messages else ""

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using only the provided context, 
            please answer the user's question in a friendly, conversational tone.
            If you don't know the answer based on the context, say you don't know.

            Context:
            {context}

            Last message provided to the user:
            {last_message}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)
        power_llm_with_tool = power_llm.bind_functions([convert_to_openai_function(send_text_tool)])
        chain = base_rag_prompt | power_llm_with_tool
        full_response = ""
        cl_msg = config["configurable"]["cl_msg"]

        async for chunk in chain.astream(
            {"question": question, "context": selected_recipe["text"], "last_message": last_message},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if isinstance(chunk, AIMessageChunk):
                await cl_msg.stream_token(chunk.content)
                full_response += chunk.content

        return {"messages": [full_response]}

    async def _node_send_sms(state: AgentState, config):
        print("--- SEND SMS ---")

        question = state["question"]
        selected_recipe = state.get("selected_recipe")
        messages = state["messages"]
        last_message = messages[-1] if messages else ""

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant.
            Using only the provided context and the tool,
            please fullfill the user's request to send an SMS text

            Context:
            {context}

            Last message provided to the user:
            {last_message}

            Question:
            {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)
        # tool_functions =
        power_llm_with_tool = power_llm.bind_functions([convert_to_openai_function(send_text_tool)])
        chain = base_rag_prompt | power_llm_with_tool

        tool_executor = ToolExecutor([send_text_tool])
        message = chain.invoke(
            {
                "question": question,
                "context": selected_recipe.get("text") if selected_recipe else "",
                "last_message": last_message,
            },
        )

        print("message", message)

        action = ToolInvocation(
            tool=message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(message.additional_kwargs["function_call"]["arguments"]),
        )

        response = tool_executor.invoke(action)

        function_message = FunctionMessage(content=str(response), name=action.tool)

        return {"messages": [function_message]}

    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("triage", _node_question_triage)  # retrieve
    workflow.add_node("retrieve", _node_call_retriever)  # retrieve
    workflow.add_node("grade_recipes", _node_grade_recipes)  # grade documents
    workflow.add_node("generate", _node_generate_response)  # generatae
    workflow.add_node("shortlist_qa", _node_shortlist_qa)  # answer questions about shortlisted recipes
    workflow.add_node("single_qa", _node_single_recipe_qa)  # answer questions about shortlisted recipes
    workflow.add_node("send_sms", _node_send_sms)  # answer questions about shortlisted recipes

    # Define the edges

    def _edge_route_question(state: AgentState):
        print("=======EDGE: START =====")
        question_type = state["question_type"]
        messages = state["messages"]
        shortlisted_recipes = state.get("shortlisted_recipes")
        selected_recipe = state.get("selected_recipe")

        # if not shortlisted_recipes or len(shortlisted_recipes) == 0:
        #     print("going to retrieve since no shortlisted_recipes")
        #     return "retrieve"
        if question_type == "asking_for_recipe_suggestions":
            return "retrieve"
        if question_type in ["referring_to_shortlisted_recipes", "show_specific_recipe"]:
            return "shortlist_qa"
        if question_type == "referring_to_specific_recipe" and selected_recipe:
            return "single_qa"
        if question_type == "send_sms":
            return "send_sms"

        print("defaulting to shortlist_qa")
        return "shortlist_qa"

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges(
        "triage",
        _edge_route_question,
        {
            "shortlist_qa": "shortlist_qa",
            "single_qa": "single_qa",
            "retrieve": "retrieve",
            "send_sms": "send_sms",
        },
    )

    workflow.add_edge("retrieve", "grade_recipes")
    workflow.add_edge("grade_recipes", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("shortlist_qa", END)
    workflow.add_edge("single_qa", END)
    workflow.add_edge("send_sms", END)

    memory = AsyncSqliteSaver.from_conn_string(":memory:")

    app = workflow.compile(checkpointer=memory)

    return app
