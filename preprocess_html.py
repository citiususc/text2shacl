import argparse
from bs4 import BeautifulSoup

def preprocess_html(html_file, output_file):
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    if soup.body:
        soup.body["class"] = soup.body.get("class", []) + ["Document"]

    def es_fila_number(fila):
        return any(td.get_text(strip=True).lower() == "number" for td in fila.find_all("td"))

    def fila_valida(fila):
        return any(td.get_text(strip=True) for td in fila.find_all("td"))

    def crear_tabla_con_borde(filas):
        tabla = soup.new_tag("table")
        for fila in filas:
            tabla.append(fila)
        return tabla

    def mover_listas_dentro_tablas(li):
        tablas = li.find_all("table")
        for tabla in tablas:
            siguiente = tabla.find_next_sibling()
            while siguiente and siguiente.name in ["ul", "ol"]:
                lista = siguiente.extract()
                filas = tabla.find_all("tr")
                if filas:
                    ultima_fila = filas[-1]
                    celdas = ultima_fila.find_all("td")
                    if celdas:
                        celdas[-1].append(lista)
                siguiente = tabla.find_next_sibling()

    def procesar_list_of_parameters(li):
        mover_listas_dentro_tablas(li)

        nuevo_contenido = []
        todas_las_filas = []

        for elem in list(li.children):
            if elem.name == "table":
                filas = [f for f in elem.find_all("tr") if fila_valida(f)]
                todas_las_filas.extend(filas)
            else:
                nuevo_contenido.append(elem)

        tablas = []
        tabla_actual = []

        for fila in todas_las_filas:
            if es_fila_number(fila):
                if tabla_actual:
                    tablas.append(tabla_actual)
                tabla_actual = [fila]
            else:
                tabla_actual.append(fila)

        if tabla_actual:
            tablas.append(tabla_actual)

        li.clear()

        for elem in nuevo_contenido:
            li.append(elem)

        for tabla_filas in tablas:
            tabla = crear_tabla_con_borde(tabla_filas)
            li.append(tabla)
            separador = soup.new_tag("hr", style="border: 0px solid black; margin: 20px 0;")
            li.append(separador)

    li_list_of_params = None
    for li in soup.find_all("li", {"data-list-text": "5.1"}):
        if li.find("h1") and "List of parameters: table 5" in li.find("h1").get_text(strip=True):
            li_list_of_params = li
            break

    if li_list_of_params:
        procesar_list_of_parameters(li_list_of_params)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(soup))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess HTML file to the partition_html() function")
    parser.add_argument("file", help="Path to the html file to be processed.")
    parser.add_argument("output", help="Path to the html output file.")

    args = parser.parse_args()

    html = preprocess_html(args.file, args.output)