import networkx as nx
import matplotlib.pyplot as plt


def leer_gramatica(archivo):
    """Lee la gramática y normaliza símbolos terminales a los nombres de token."""
    mapeo = {
        "(": "pari", ")": "pard",
        "+": "opsuma", "-": "opsuma",
        "*": "opmul", "/": "opmul"
    }
    gramatica = {}
    with open(archivo, "r", encoding="utf-8") as f:
        for lineno, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "->" not in linea:
                raise ValueError(f"Línea {lineno} inválida en la gramática (falta '->'): {linea}")
            izquierda, derecha = linea.split("->", 1)
            izquierda = izquierda.strip()
            alternativas = []
            for alt in derecha.strip().split("|"):
                simbolos = [s.strip() for s in alt.strip().split() if s.strip()]
                # normalizar paréntesis u otros símbolos si el autor usó '(' en lugar de 'pari', etc.
                simbolos_norm = [mapeo.get(s, s) for s in simbolos]
                alternativas.append(simbolos_norm)
            gramatica.setdefault(izquierda, []).extend(alternativas)
    return gramatica


def tokenizar(expr):
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isdigit():
            num = c
            i += 1
            while i < len(expr) and expr[i].isdigit():
                num += expr[i]; i += 1
            tokens.append(("num", num))
        elif c.isalpha():
            ident = c
            i += 1
            while i < len(expr) and expr[i].isalnum():
                ident += expr[i]; i += 1
            tokens.append(("id", ident))
        elif c in "+-":
            tokens.append(("opsuma", c)); i += 1
        elif c in "*/":
            tokens.append(("opmul", c)); i += 1
        elif c == "(":
            tokens.append(("pari", c)); i += 1
        elif c == ")":
            tokens.append(("pard", c)); i += 1
        elif c.isspace():
            i += 1
        else:
            raise ValueError(f"Caracter inválido: {c}")
    return tokens


class Validador:
    def __init__(self, grammar):
        self.grammar = grammar

    def parse(self, tokens, start="E"):
        self.tokens = tokens
        self.chart = [set() for _ in range(len(tokens) + 1)]
        # estado inicial: considerar *todas* las producciones del start está implícito
        # añadimos un estado por cada producción del símbolo inicial
        for prod in self.grammar.get(start, []):
            self.chart[0].add((start, tuple(prod), 0, 0))

        for i in range(len(tokens) + 1):
            changed = True
            while changed:
                changed = False
                for state in list(self.chart[i]):
                    lhs, rhs, dot, origin = state
                    # si hay símbolo después del punto
                    if dot < len(rhs):
                        sym = rhs[dot]
                        # Predictor: si es no terminal
                        if sym in self.grammar:
                            for prod in self.grammar[sym]:
                                new_state = (sym, tuple(prod), 0, i)
                                if new_state not in self.chart[i]:
                                    self.chart[i].add(new_state); changed = True
                        # Scanner: si es terminal y hay token en i
                        elif i < len(tokens):
                            token_type, _ = tokens[i]
                            if sym == token_type:
                                new_state = (lhs, rhs, dot + 1, origin)
                                # añadir al siguiente conjunto
                                if new_state not in self.chart[i + 1]:
                                    self.chart[i + 1].add(new_state); changed = True
                    else:
                        # Completer: intento de completar estados que esperaban 'lhs'
                        for prev in list(self.chart[origin]):
                            plhs, prhs, pdot, porigin = prev
                            if pdot < len(prhs) and prhs[pdot] == lhs:
                                new_state = (plhs, prhs, pdot + 1, porigin)
                                if new_state not in self.chart[i]:
                                    self.chart[i].add(new_state); changed = True

        # aceptación: existe un estado completado para el símbolo inicial con origen 0
        for st in self.chart[len(tokens)]:
            if st[0] == start and st[2] == len(st[1]) and st[3] == 0:
                return True
        return False


class Nodo:
    def __init__(self, etiqueta, hijos=None):
        self.etiqueta = etiqueta
        self.hijos = hijos if hijos else []


def construir_arbol_sintactico(tokens):
    """Construcción por descenso recursivo (corresponde a la gramática E/T/F usada)."""
    def parse_E(i):
        nodo_T, i = parse_T(i)
        if i < len(tokens) and tokens[i][0] == "opsuma":
            op = tokens[i][1]
            nodo_E, j = parse_E(i + 1)
            return Nodo("E", [nodo_T, Nodo("opsuma", [Nodo(op)]), nodo_E]), j
        return Nodo("E", [nodo_T]), i

    def parse_T(i):
        nodo_F, i = parse_F(i)
        if i < len(tokens) and tokens[i][0] == "opmul":
            op = tokens[i][1]
            nodo_T, j = parse_T(i + 1)
            return Nodo("T", [nodo_F, Nodo("opmul", [Nodo(op)]), nodo_T]), j
        return Nodo("T", [nodo_F]), i

    def parse_F(i):
        if i >= len(tokens):
            raise ValueError("Token inesperado: fin de entrada en F")
        tok_type, tok_val = tokens[i]
        if tok_type == "num":
            return Nodo("F", [Nodo("num", [Nodo(tok_val)])]), i + 1
        elif tok_type == "id":
            return Nodo("F", [Nodo("id", [Nodo(tok_val)])]), i + 1
        elif tok_type == "pari":
            nodo_E, j = parse_E(i + 1)
            if j >= len(tokens) or tokens[j][0] != "pard":
                raise ValueError("Faltó cerrar paréntesis")
            return Nodo("F", [Nodo("pari", [Nodo("(")]), nodo_E, Nodo("pard", [Nodo(")")])]), j + 1
        else:
            raise ValueError(f"Token inesperado en F: {tokens[i]}")

    raiz, pos_final = parse_E(0)
    # opcional: comprobar que se consumió toda la entrada
    if pos_final != len(tokens):
        raise ValueError("No se pudo consumir toda la entrada al construir el árbol")
    return raiz


def agregar_nodos_edges(G, nodo, parent=None, contador=None):
    if contador is None:
        contador = [0]
    idx = contador[0]; contador[0] += 1
    G.add_node(idx, label=nodo.etiqueta)
    if parent is not None:
        G.add_edge(parent, idx)
    for hijo in nodo.hijos:
        agregar_nodos_edges(G, hijo, idx, contador)
    return G


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {}
    def _hierarchy_pos(G, node, left, right, vert_loc, pos):
        pos[node] = ((left + right) / 2, vert_loc)
        children = list(G.successors(node))
        if children:
            dx = (right - left) / len(children)
            nextx = left
            for child in children:
                _left = nextx
                nextx += dx
                _hierarchy_pos(G, child, _left, nextx, vert_loc - vert_gap, pos)
    _hierarchy_pos(G, root, 0, width, vert_loc, pos)
    return pos


def dibujar_arbol_sintactico(raiz):
    G = nx.DiGraph()
    G = agregar_nodos_edges(G, raiz)
    labels = nx.get_node_attributes(G, "label")
    root = 0  # por construcción, el primer nodo es la raíz
    pos = hierarchy_pos(G, root)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, labels=labels, with_labels=True,
            node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold")
    plt.show()


if __name__ == "__main__":
    gramatica = leer_gramatica("gra.txt")

    with open("cad.txt", "r", encoding="utf-8") as f:
        expr = f.read().strip()

    tokens = tokenizar(expr)

    parser = Validador(gramatica)
    valido = parser.parse(tokens)

    if valido:
        print("Cadena valida")
        raiz = construir_arbol_sintactico(tokens)
        dibujar_arbol_sintactico(raiz)
    else:
        print("No se reconoce la cadena")


