import json
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition, ToolExecutor, ToolInvocation
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import StateGraph, END

from .retrievers import get_self_retriever



class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]



def get_tool_belt(base_llm):
    retriever = get_self_retriever(base_llm)

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_recipes",
        "Searches library of available recipes based on user criteria",
    )

    tool_belt = [retriever_tool]

    return tool_belt

def get_llm_model_with_functions(base_llm, tool_belt):
    functions = [convert_to_openai_function(t) for t in tool_belt]
    model_with_functions = base_llm.bind_functions(functions)
    return model_with_functions


def generate_workflow(base_llm):
    tool_belt = get_tool_belt(base_llm)
    llm_model_with_functions = get_llm_model_with_functions(base_llm, tool_belt)



    def _node_call_model(state):
        messages = state["messages"]
        response = llm_model_with_functions.invoke(messages)
        return {"messages" : [response]}

    def _node_call_tool(state):
        last_message = state["messages"][-1]

        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            )
        )
        tool_executor = ToolExecutor(tool_belt)
        response = tool_executor.invoke(action)

        print("rag retreival output:", response)
        print(type(response))

        function_message = FunctionMessage(content=str(response), name=action.tool)

        return {"messages" : [function_message]}

    def _edge_should_continue(state):
        last_message = state["messages"][-1]

        if "function_call" not in last_message.additional_kwargs:
            return END

        return "continue"

    def _node_evaluate_recipes(state):
        """
        Determines whether the retrieved recipes are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECKING RECIPE RELEVANCE---")

        # LLM with tool and validation
        base_rag_prompt_template = """\
            You are a friendly AI assistant. Using the provided context, please answer the user's question in a friendly, conversational tone. 

            Based on the context provided, please select the top 3 receipes that best fits criteria outlined in the question and is most suitable given any other requirements in the question. 
            
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
        messages = state["messages"]

        user_question = messages[0].content
        context = messages[-1].content
        
        chain = base_rag_prompt | base_llm
        response = chain.invoke({"question": user_question, "context": context})

        return {"messages": [response]}


    workflow = StateGraph(AgentState)

    workflow.add_node("agent", _node_call_model)
    workflow.add_node("action", _node_call_tool)
    workflow.add_node("evaluate", _node_evaluate_recipes)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        _edge_should_continue,
        {
            "continue" : "action",
            END : END
        }
    )
    workflow.add_edge("action", "evaluate")
    workflow.add_edge("evaluate", "agent")

    app = workflow.compile()

    return app



def log_state_messages(messages):
    next_is_tool = False
    initial_query = True
    for message in messages["messages"]:
        print("===========")
        print(message.__dict__.keys())
        print("message.name:", message.name)
        print("message.type:", message.type)
        print("message.response_metadata:", message.response_metadata)
        print('message.additional_kwargs:', message.additional_kwargs)
        # if "function_call" in message.additional_kwargs:
        #     print()
        #     print(f'Tool Call - Name: {message.additional_kwargs["function_call"]["name"]} + Query: {message.additional_kwargs["function_call"]["arguments"]}')
        #     next_is_tool = True
        #     continue
        # if next_is_tool:
        #     print(f"Tool Response: {message.content}")
        #     next_is_tool = False
        #     continue
        # if initial_query:
        #     print(f"Initial Query: {message.content}")
        #     print()
        #     initial_query = False
        #     continue
        # print()
        # print(f"Agent Response: {message.content}")