from rdflib import Graph, RDF, RDFS, OWL

if __name__ == "__main__":
    # Load the ontology from a TTL file
    g = Graph()
    g.parse("https://data-interop.era.europa.eu/era-vocabulary/ontology.ttl", format="turtle")

    # Query all classes in the ontology
    query_classes = """
        SELECT ?class ?comment WHERE {
            ?class a owl:Class .
            FILTER(!isBlank(?class))  # Excluye blank nodes
            OPTIONAL { ?class rdfs:comment ?comment . }
            FILTER (lang(?comment) = "en" || !BOUND(?comment))  # Solo comentarios en inglés o sin comentario
        }
    """

    class_info = []
    for row in g.query(query_classes):
        class_uri = row[0]
        comment = row[1] if row[1] else "No description available"
        class_info.append({"name": str(class_uri), "description": str(comment)})


    # Run SHACL shape generation for each entity
    for entity in class_info:
        print(entity['name'])