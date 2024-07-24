from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from .db import shortlisted_recipes_to_string


def get_grader_chain(llm_model):
    class GradeRecipes(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Document representing recipes are generally relevant to the criteria in the question, 'yes' or 'no'"
        )

        integer_score: int = Field(
            description="Degree to which Documents are relevant to the question, integers from 1 to 100"
        )

    # LLM with function call
    structured_llm_grader = llm_model.with_structured_output(GradeRecipes)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved cooking recipe document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out completely erroneous or irrelevant retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the recipe document is relevant to the question.
        Also give a integer score from 1 to 100 to indicate the degree to which the recipe document is relevant to the question.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved recipe document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader


def get_recipe_url_extractor_chain(llm_model):
    class RecipeUrlsSchema(BaseModel):
        urls: list[str] = Field(description="A list of urls pointing to specific recipes")

    structured_llm_grader = llm_model.with_structured_output(RecipeUrlsSchema)

    pydantic_parser = PydanticOutputParser(pydantic_object=RecipeUrlsSchema)
    format_instructions = pydantic_parser.get_format_instructions()

    RECIPE_SEARCH_PROMPT = """
        Your goal is to understand and parse out the full http urls in the context corresponding to each recipe.

        {format_instructions}

        Context:
        {context}
    """

    prompt = ChatPromptTemplate.from_template(
        template=RECIPE_SEARCH_PROMPT, partial_variables={"format_instructions": format_instructions}
    )

    retriever = prompt | structured_llm_grader

    return retriever


def get_recipe_selection_chain(llm_model):
    class RecipeSelectionSchema(BaseModel):
        asking_for_recipe_suggestions: str = Field(
            description="Whether the User Question is asking for recipe suggestions based on some criteria, 'yes' or 'no'"
        )
        referring_to_specific_recipe: str = Field(
            description="Whether the User Question is asking about one specific recipe (but NOT asking to just show a specific recipe), 'yes' or 'no'"
        )
        referring_to_shortlisted_recipes: str = Field(
            description="Whether the User Question is asking generally about the 3 shortlisted recipes, 'yes' or 'no'"
        )

        show_specific_recipe: str = Field(
            description="Whether the User Question is asking asking to show a specific recipe, 'yes' or 'no'"
        )

        specific_recipe_url: str = Field(
            description="URL of the specific recipe that the User Question is directed to, if any "
        )

    # LLM with function call
    structured_llm_grader = llm_model.with_structured_output(RecipeSelectionSchema)
    pydantic_parser = PydanticOutputParser(pydantic_object=RecipeSelectionSchema)
    format_instructions = pydantic_parser.get_format_instructions()

    # Prompt
    RECIPE_SELECTION_PROMPT = """
        You are a helpful assistant attempting to categorize the nature of the User question
        based on the last message sent to he user and the provided context.

        {format_instructions}
        
        User Question:
        {question}

        Last message provided to the user:
        {last_message}

        Context:
        {context}

        
    """

    prompt = ChatPromptTemplate.from_template(
        template=RECIPE_SELECTION_PROMPT, partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | structured_llm_grader

    return chain


def get_question_type_chain(llm_model):
    class RecipeSelectionChanceSchema(BaseModel):
        asking_for_recipe_suggestions: int = Field(
            description="The likelihood / chance that the User Question is asking for recipe suggestions based on some criteria, integers from 1 to 100"
        )
        referring_to_specific_recipe: int = Field(
            description="The likelihood / chance that the User Question is asking specific questions about a single specific recipe, integers from 1 to 100"
        )
        referring_to_shortlisted_recipes: int = Field(
            description="The likelihood / chance that the User Question is asking generally about more than one recipe provided in the last message, integers from 1 to 100"
        )

        show_specific_recipe: int = Field(
            description="The likelihood / chance that the User Question is asking to show the full recipe for a specific recipe, integers from 1 to 100"
        )
        send_text: int = Field(
            description="The likelihood / chance that the User Question is to send a SMS or text, integers from 1 to 100"
        )

        specific_recipe_url: str = Field(
            description="URL of the specific recipe that the User Question is directed to, if any "
        )

    # LLM with function call
    structured_llm_grader = llm_model.with_structured_output(RecipeSelectionChanceSchema)
    pydantic_parser = PydanticOutputParser(pydantic_object=RecipeSelectionChanceSchema)
    format_instructions = pydantic_parser.get_format_instructions()

    # Prompt
    RECIPE_SELECTION_PROMPT = """
        You are a helpful assistant attempting to categorize the nature of the User question
        based on the last message sent to he user and the provided context. 
        Note that if there were recipe suggesetions in the last message provided to the user, 
        it is highly likely that the user is asking questions referring to shortlisted recipes.
        If the last message was a full single recipe, it is generally likely that the user 
        is asking questions referring to specific recipe.
        If the user is asking to show the full recipe, it is highly likely that they are asking 
        to show a specific recipe and less likely that they are asking for anything else.

        {format_instructions}
        
        User Question:
        {question}

        Last message provided to the user:
        {last_message}

        Context:
        {context}

        
    """

    prompt = ChatPromptTemplate.from_template(
        template=RECIPE_SELECTION_PROMPT, partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | structured_llm_grader

    return chain


def get_selected_recipe(llm_model, question, shortlisted_recipes, messages):
    selected_recipe = None
    recipe_selection_chain = get_recipe_selection_chain(llm_model)
    recipe_selection_response = recipe_selection_chain.invoke(
        {
            "question": question,
            "context": shortlisted_recipes_to_string(shortlisted_recipes),
            "last_message": messages[-1] if messages else "",
        }
    )

    if (
        recipe_selection_response.referring_to_specific_recipe == "yes"
        and recipe_selection_response.specific_recipe_url
    ):
        selected_recipe = next(
            (r for r in shortlisted_recipes if r["url"] == recipe_selection_response.specific_recipe_url)
        )
    return selected_recipe
