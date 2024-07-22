from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


def get_grader_chain(llm_model):

    class GradeRecipes(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

        integer_score: int = Field(
            description="Degree to which Documents are relevant to the question, integers from 1 to 100"
        )

    # LLM with function call
    structured_llm_grader = llm_model.with_structured_output(GradeRecipes)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved cooking recipe document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous or irrelevant retrievals. \n
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
        template=RECIPE_SEARCH_PROMPT,
        partial_variables = {
            "format_instructions": format_instructions 
        }
    )

    retriever = prompt | structured_llm_grader

    return retriever



def get_recipe_selection_chain(llm_model):

    class RecipeSelectionSchema(BaseModel):
        asking_for_recipe_suggestions: str = Field(
            description="Whether the User Question is either asking for recipe suggestions, 'yes' or 'no'"
        )
        referring_to_specific_recipe: str = Field(
            description="Whether the User Question is asking about one specific recipe, 'yes' or 'no'"
        )
        referring_to_shortlisted_recipes: str = Field(
            description="Whether the User Question is asking generally about the 3 shortlisted recipes, 'yes' or 'no'"
        )

        # specific_recipe: str = Field(
        #     description="URL of the specific recipe that the User Question is directed to, if any "
        # )
    # LLM with function call
    structured_llm_grader = llm_model.with_structured_output(RecipeSelectionSchema)
    pydantic_parser = PydanticOutputParser(pydantic_object=RecipeSelectionSchema)
    format_instructions = pydantic_parser.get_format_instructions()
    
    # Prompt
    RECIPE_SELECTION_PROMPT = """
        You are a helpful assistant attempting to determine the nature of the User question. 

        {format_instructions}
        
        User Question:
        {question}

        Context:
        {context}
    """

    prompt = ChatPromptTemplate.from_template(
        template=RECIPE_SELECTION_PROMPT,
        partial_variables = {
            "format_instructions": format_instructions 
        }
    )

    chain = prompt | structured_llm_grader

    return chain