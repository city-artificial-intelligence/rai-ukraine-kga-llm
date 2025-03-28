# ruff: noqa: E501
from src.onto_object import OntologyEntryAttr


def prompt_only_names(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}

    Response with True or False
    """


def prompt_with_hierarchy(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Parent Entity Namings: {src_entety.get_parents_preferred_names()}
    Child Entity Namings: {src_entety.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}
    Child Entity Namings: {tgt_entety.get_children_preferred_names()}

    Response with True or False
    """


def prompt_only_with_parents(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and parent relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Parent Entity Namings: {src_entety.get_parents_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}

    Response with True or False
    """


def prompt_only_with_children(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Child Entity Namings: {src_entety.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Child Entity Namings: {tgt_entety.get_children_preferred_names()}

    Response with True or False
    """
