import os
from supabase import Client, create_client

SUPABASE_URL: str = "https://cfivdlyedzbcvjsztebc.supabase.co"
SUPABASE_SECRET_KEY: str = os.environ["SUPABASE_SECRET_KEY"]


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def get_recipes(urls: list):
    supabase_client = get_client()
    response = (
        supabase_client.table("recipes")
        .select("url, metadata, features, md_ingredients, md_preparation, md_nutrition, md_description, time")
        .in_("url", urls)
        .limit(10)
        .execute()
    )

    return [
        {
            "title": r["metadata"]["title"],
            "thumbnail": r["metadata"].get("thumbnail"),
            "url": r["url"],
            "text": f"""
            TITLE: \n
            {r['metadata']['title']}
            \n\n
            ESTIMATED COOKING / PREPARATION TIME: {r['time']} minutes
            \n\n
            DESCRIPTION: \n
            {r['md_description']}
            \n\n
            INGREDIENTS: \n
            {r['md_ingredients']}
            NUTRITIONAL INFORMATION: \n
            {r['md_nutrition']}
            \n\n
            PREP INSTRUCTIONS: \n
            {r['md_preparation']}

            Source URL: {r['url']}
            \n\n
        """,
        }
        for r in response.data
    ]


def shortlisted_recipes_to_string(recipes):
    output = ""
    if recipes and isinstance(recipes, list):
        for index, r in enumerate(recipes):
            output += f"""Suggestion #{index+1}: {r['text']} \n\n"""
    return output
