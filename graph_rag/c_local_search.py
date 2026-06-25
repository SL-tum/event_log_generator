from collections.abc import AsyncGenerator
from typing import Any
import pandas as pd
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig

from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import (
    get_embedding_store,
    load_search_prompt,
)

def local_search_streaming(
    root_folder_path: str,
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,
    relationships: pd.DataFrame,
    covariates: pd.DataFrame | None,
    community_level: int,
    response_type: str,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
) -> AsyncGenerator[str, None]:
    
    vector_store_args = config.vector_store
    
    embedding_name = "entity_description" 

    description_embedding_store = get_embedding_store(
        vector_store_args,
        embedding_name,
    )

    entities_ = read_indexer_entities(entities, communities, community_level)
    covariates_ = read_indexer_covariates(covariates) if covariates is not None else []

    root_dir = f"{root_folder_path}/{config.local_search.prompt}"
    prompt = load_search_prompt(root_dir)

    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(community_reports, communities, community_level),
        text_units=read_indexer_text_units(text_units),
        entities=entities_,
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
        system_prompt=prompt,
        callbacks=callbacks,
    )

    return search_engine.stream_search(query=query)

async def local_search(
    root_folder_path: str,
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,
    relationships: pd.DataFrame,
    community_level: int,
    response_type: str,
    query: str,
    covariates: pd.DataFrame | None = None,
    callbacks: list[QueryCallbacks] | None = None,
) -> tuple[str, dict[str, Any]]:
    
    callbacks = callbacks or []
    context_data = {}

    def on_context(context: Any) -> None:
        nonlocal context_data
        context_data = context

        context_data["reports"].to_csv("data_output.csv", index=False, encoding="utf-8-sig")
        #print(f"this is context data: {context_data}")
        #with open("context.txt", "w", encoding = "utf=8") as f:
        #    f.write("\n".join(context_data["reports"]['context'].astype(str)))
        #print("Context logic triggered")

    local_callbacks = NoopQueryCallbacks()
    local_callbacks.on_context = on_context
    callbacks.append(local_callbacks)

    full_response_list = []
    
    async for chunk in local_search_streaming(
        root_folder_path= root_folder_path,
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=covariates,
        community_level=community_level,
        response_type=response_type,
        query=query,
        callbacks=callbacks,
    ):
        full_response_list.append(str(chunk))
    
    return "".join(full_response_list), context_data

