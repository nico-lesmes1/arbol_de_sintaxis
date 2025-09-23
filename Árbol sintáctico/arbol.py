import networkx as nx
import matplotlib.pyplot as plt


def leer_gramatica(archivo):
    gramatica = {}
    with open(archivo, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            izquierda, derecha = linea.split("->")
            izquierda = izquierda.strip()
            producciones = [r.strip().split() for r in derecha.strip().split("|")]
            if izquierda not in gramatica:
                gramatica[izquierda] = []
            gramatica[izquierda].extend(producciones)
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
                num += expr[i]
                i += 1
            tokens.append(("num", num))
        elif c.isalpha():
            tokens.append(("id", c))
            i += 1
        elif c in "+-":
            tokens.append(("opsuma", c))
            i += 1
        elif c in "*/":
            tokens.append(("opmul", c))
            i += 1
        elif c == "(":
            tokens.append(("pari", c))
            i += 1
        elif c == ")":
            tokens.append(("pard", c))
            i += 1
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
        self.chart = [set() for _ in range(len(tokens)+1)]
        self.chart[0].add((start, tuple(self.grammar[start][0]), 0, 0))

        for i in range(len(tokens)+1):
            changed = True
            while changed:
                changed = False
                for state in list(self.chart[i]):
                    lhs, rhs, dot, start_index = state
                    if dot < len(rhs) and rhs[dot] in self.grammar:
                        for prod in self.grammar[rhs[dot]]:
                            new_state = (rhs[dot], tuple(prod), 0, i)
                            if new_state not in self.chart[i]:
                                self.chart[i].add(new_state)
                                changed = True
                    elif dot < len(rhs) and i < len(tokens):
                        token_type, _ = tokens[i]
                        if rhs[dot] == token_type:
                            new_state = (lhs, rhs, dot+1, start_index)
                            self.chart[i+1].add(new_state)
                    elif dot == len(rhs):
                        for prev in list(self.chart[start_index]):
                            plhs, prhs, pdot, pstart = prev
                            if pdot < len(prhs) and prhs[pdot] == lhs:
                                new_state = (plhs, prhs, pdot+1, pstart)
                                if new_state not in self.chart[i]:
                                    self.chart[i].add(new_state)
                                    changed = True

        for st in self.chart[len(tokens)]:
            if st[0] == start and st[1] == tuple(self.grammar[start][0]) and st[2] == len(st[1]) and st[3] == 0:
                return True
        return False

class Nodo:
    def __init__(self, etiqueta, hijos=None):
        self.etiqueta = etiqueta
        self.hijos = hijos if hijos else []

def construir_arbol_sintactico(tokens):
    """ Construye el árbol sintáctico completo """

    def parse_E(i):
        nodo_T, i = parse_T(i)
        if i < len(tokens) and tokens[i][0] == "opsuma":
            op = tokens[i][1]
            nodo_E, j = parse_E(i+1)
            return Nodo("E", [nodo_T, Nodo("opsuma", [Nodo(op)]), nodo_E]), j
        return Nodo("E", [nodo_T]), i

    def parse_T(i):
        nodo_F, i = parse_F(i)
        if i < len(tokens) and tokens[i][0] == "opmul":
            op = tokens[i][1]
            nodo_T, j = parse_T(i+1)
            return Nodo("T", [nodo_F, Nodo("opmul", [Nodo(op)]), nodo_T]), j
        return Nodo("T", [nodo_F]), i

    def parse_F(i):
        tok_type, tok_val = tokens[i]
        if tok_type == "num":
            return Nodo("F", [Nodo("num", [Nodo(tok_val)])]), i+1
        elif tok_type == "id":
            return Nodo("F", [Nodo("id", [Nodo(tok_val)])]), i+1
        elif tok_type == "pari":
            nodo_E, j = parse_E(i+1)
            return Nodo("F", [Nodo("pari", [Nodo("(")]), nodo_E, Nodo("pard", [Nodo(")")])]), j+1
        raise ValueError(f"Token inesperado en F: {tokens[i]}")

    raiz, _ = parse_E(0)
    return raiz

def agregar_nodos_edges(G, nodo, parent=None, contador=[0]):
    idx = contador[0]
    contador[0] += 1
    G.add_node(idx, label=nodo.etiqueta)
    if parent is not None:
        G.add_edge(parent, idx)
    for hijo in nodo.hijos:
        agregar_nodos_edges(G, hijo, idx, contador)
    return G

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {}
    def _hierarchy_pos(G, node, left, right, vert_loc, pos):
        pos[node] = ((left+right)/2, vert_loc)
        children = list(G.successors(node))
        if len(children) != 0:
            dx = (right-left)/len(children)
            nextx = left
            for child in children:
                nextx += dx
                _hierarchy_pos(G, child, nextx-dx, nextx, vert_loc-vert_gap, pos)
    _hierarchy_pos(G, root, 0, width, vert_loc, pos)
    return pos

def dibujar_arbol_sintactico(raiz):
    G = nx.DiGraph()
    G = agregar_nodos_edges(G, raiz)

    labels = nx.get_node_attributes(G, "label")
    pos = hierarchy_pos(G, 0)

    plt.figure(figsize=(10,6))
    nx.draw(G, pos, labels=labels, with_labels=True,
            node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold")
    plt.show()


if __name__ == "__main__":
    gramatica = leer_gramatica("gra.txt")

    with open("cad.txt") as f:
        expr = f.read().strip()

    tokens = tokenizar(expr)

    parser = Validador(gramatica)
    valido = parser.parse(tokens)

    print("Tokens:", tokens)
    print("Cadena valida:", valido)

    if valido:
        raiz = construir_arbol_sintactico(tokens)
        dibujar_arbol_sintactico(raiz)
