from typing import Dict, List
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, BNode
from rdflib.namespace import SH
from urllib.parse import urlparse
import re

# --- AUXILIARY FUNCTIONS ---

def process_shacl(shacl_text: str, output_path: str):
    """
    Cleans a SHACL Turtle file by removing duplicate prefix declarations
    while preserving the order of the first occurrence, and writes
    the cleaned content to the specified output path.
    """
    # Pattern to match prefix lines
    prefix_pattern = re.compile(r'^@prefix\s+[^:]+:\s+<[^>]+>\s+\.', re.MULTILINE)
    
    # Extract all prefix definitions
    all_prefixes = prefix_pattern.findall(shacl_text)
    
    # Remove duplicates while keeping the first occurrence
    seen = set()
    unique_prefixes = []
    for p in all_prefixes:
        if p not in seen:
            unique_prefixes.append(p)
            seen.add(p)
    
    # Remove all prefix lines from the original text
    text_without_prefixes = prefix_pattern.sub('', shacl_text).strip()
    
    # Build the final cleaned text
    shacl_cleaned = '\n'.join(unique_prefixes) + '\n\n' + text_without_prefixes
    
    # Write to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shacl_cleaned)


def generate_node_shapes_str(node_shapes: dict) -> str:
    """
    Generates SHACL NodeShape definitions as Turtle text
    based on a dictionary of class qualified names and their properties.
    """
    shapes_str = ""
    for class_qname, properties in node_shapes.items():
        shape_name = f"{class_qname}Shape"
        properties_str = ",\n               ".join(properties)
        
        shape_block = f"""{shape_name} a sh:NodeShape ;
               sh:property {properties_str} ;
               sh:targetClass {class_qname} .\n\n"""
        
        shapes_str += shape_block

    return shapes_str.strip()


def update_node_shapes(node_shapes: Dict[str, List[str]], affected_classes: List[str], shape_name: str):
    """
    Updates a dictionary of node shapes by adding a new shape name
    to all affected classes, creating the class key if it does not exist.
    """
    for affected_class in affected_classes:
        if affected_class not in node_shapes:
            node_shapes[affected_class] = []
        if shape_name not in node_shapes[affected_class]:
            node_shapes[affected_class].append(shape_name)
    return node_shapes


def extract_name_and_class_from_shape(shacl_str: str, shacl_prefixes: str):
    """
    Extracts the qualified name of a PropertyShape and its associated affectedClass
    values from a SHACL Turtle fragment.

    :param shacl_str: SHACL fragment (single PropertyShape) as a string
    :param shacl_prefixes: Required Turtle-style prefixes as a string
    :return: Tuple (property_shape_qname: str, affected_classes_qnames: List[str])
    """
    g = Graph()
    g.parse(data=f"{shacl_prefixes}\n{shacl_str}", format="turtle")

    ERA = Namespace("http://data.europa.eu/949/")

    for s in g.subjects(RDF.type, SH.PropertyShape):
        try:
            shape_qname = g.qname(s)
        except:
            shape_qname = str(s)

        # Collect all affectedClass values
        affected_classes = []
        for c in g.objects(s, ERA.affectedClass):
            try:
                class_qname = g.qname(c)
            except:
                class_qname = str(c)
            affected_classes.append(class_qname)

        return shape_qname, affected_classes

    return None, []


def get_entity_label_and_description(g: Graph, entity_uri: str) -> str:
    """
    Retrieves a string describing an ontology entity with its local name and
    the first available rdfs:comment.

    Example: "ContactLineSystem. A system used for supplying electrical energy..."

    :param g: RDF graph containing the ontology
    :param entity_uri: Full URI of the entity
    :return: Formatted string "LocalName. Description"
    """
    entity_ref = URIRef(entity_uri)

    # Extract the local name from the URI
    parsed = urlparse(entity_uri)
    local_name = entity_uri.rsplit('/')[-1] if '/' in parsed.path else entity_uri.rsplit('#', 1)[-1]

    # Get the first available rdfs:comment
    comments = list(g.objects(entity_ref, RDFS.comment))
    description = str(comments[0]) if comments else " "

    return f"{local_name}. {description}" if comments else f"{local_name}"


def get_properties_for_class(g: Graph, class_uri: str):
    """
    Returns all properties having the given class as their domain,
    including those within owl:unionOf definitions.
    """
    properties = set()
    class_ref = URIRef(class_uri)

    # Properties with a direct domain matching the class
    for prop in g.subjects(RDFS.domain, class_ref):
        properties.add(str(prop))

    # Properties with a domain as a BNode including owl:unionOf containing the class
    for prop, domain in g.subject_objects(RDFS.domain):
        if (domain, RDF.type, OWL.Class) in g:
            for union in g.objects(domain, OWL.unionOf):
                for member in g.items(union):
                    if member == class_ref:
                        properties.add(str(prop))

    return list(properties)


def get_property_domain(g: Graph, property_uri: str):
    """
    Returns the domain(s) of a given property in the ontology.
    If the domain is defined as an owl:unionOf, returns all members
    of the union, otherwise returns the domain directly.

    :param property_uri: Full URI of the property
    :return: List of domain URIs as strings
    """
    def expand_domain_node(g, domain_node):
        if isinstance(domain_node, URIRef):
            return [str(domain_node)]
        elif isinstance(domain_node, BNode):
            union_list = list(g.objects(domain_node, OWL.unionOf))
            if union_list:
                rdf_list_root = union_list[0]
                return [str(item) for item in g.items(rdf_list_root)]
            else:
                return []
        else:
            return []

    property_ref = URIRef(property_uri)
    raw_domains = list(g.objects(property_ref, RDFS.domain))

    expanded_domains = []
    for d in raw_domains:
        expanded_domains.extend(expand_domain_node(g, d))

    return expanded_domains


def _list2markdown(data):
    """
    Converts a list of triples (subject, predicate, object) represented
    as dictionaries into a Markdown table for better readability.
    """
    table = "| Subject | Predicate | Object |\n"
    table += "|---------|-----------|--------|\n"
    
    for entry in data:
        subject = entry['subject']
        predicate = entry['predicate']
        obj = entry['object']
        table += f"| {subject} | {predicate} | {obj} |\n"
    
    return table


def get_info_by_name(g: Graph, name: str):
    """
    Searches for an entity by its local name in the RDF graph and
    returns all associated predicates and objects in a Markdown table.

    :param g: RDF graph
    :param name: Local name of the entity
    :return: Markdown table as a string, or None if not found
    """
    subj = None
    for subject in g.subjects():
        if subject.endswith(name):
            subj = subject
            break
    
    if subj is None:
        return None
    
    result = []
    for pred, obj in g.predicate_objects(subj):
        result.append({
            "subject": str(subj),
            "predicate": str(pred),
            "object": str(obj)
        })

    markdown_info = _list2markdown(result)
    return markdown_info


def get_owl_properties_with_domain(g: Graph, namespace="http://data.europa.eu/949/"):
    """
    Returns the URIs of all OWL ObjectProperty or DatatypeProperty elements
    which have a domain defined, filtering out blank nodes that do not
    define an owl:unionOf and ignoring domains outside the given namespace.
    """
    query = """
        SELECT ?prop ?domain WHERE {
            ?prop a ?type .
            FILTER(?type IN (owl:ObjectProperty, owl:DatatypeProperty)) .
            FILTER(!isBlank(?prop)) .
            ?prop rdfs:domain ?domain .
        }
    """

    properties_with_domain = []

    for row in g.query(query, initNs={"owl": OWL, "rdfs": RDFS}):
        prop = row[0]
        domain = row[1]

        # Skip blank domains without owl:unionOf
        if isinstance(domain, BNode):
            if not any(g.objects(domain, OWL.unionOf)):
                continue
        else:
            # Ignore domains outside the target namespace
            if not str(domain).startswith(namespace):
                continue

        properties_with_domain.append(str(prop))

    return properties_with_domain
