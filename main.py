from argparse import ArgumentParser
from dataclasses import dataclass, field
from queue import PriorityQueue
import math
import os
import time

processed_nodes = 0

def string_remove_at(i, s):
    '''
    functia primeste un index (i) si un string (s) si returneaza un string
    format din caracterele lui s prin eliminarea caracterului
    de la pozitia i
    '''
    return s[:i] + s[i+1:]

def is_subseq(x, y):
    '''
    functia determina daca x este subsir al lui y
    '''
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)

@dataclass(order=True)
class Node:

    '''
    clasa Node tine informatiile legate de o configuratie de de board,
    precum starea, costul, parintele ei, liniile/ coloanele taiate
    inainte de a se ajunge aici
    '''

    def __init__(self, config, cost, prev_board, cut_rows, cut_columns):
        '''
        in constructor setam niste atribute de baza ale clasei
        '''
        self.board = config
        self.cost = cost
        self.prev_board = prev_board
        self.cut_rows = cut_rows
        self.cut_columns = cut_columns

    def get_tuple_board(self):
        '''
        functia este utila pentru a compara cu usurinta
        doua configuratii intre ele
        '''
        return tuple(self.board)

    def backtracking(self, index, prev_last, n, crt_sol, sol_list, start_time, timeout):
        '''
        este o functie basic de backtracking care ne returneaza combinari de
        numere cae vor fi folosite pentru a incerca diferite taieturi
        asupra unui board
        '''
        if crt_sol != []:
            sol_list.append(crt_sol)

        if time.time() - start_time > timeout:
            return

        if index < n:
            for i in range(prev_last + 1, n):
                new_sol = [x for x in crt_sol]
                new_sol.append(i)
                self.backtracking(index + 1, i, n, new_sol, sol_list, start_time, timeout)

    def column_differences(self, cut, configuration):
        result = 0
        for i in range(0, len(self.board) - 1):
            if self.board[i][cut] != self.board[i + 1][cut]:
                result = result + 1
            if cut + 1 in configuration:
                if self.board[i][cut] != self.board[i][cut + 1]:
                    result = result + 1

        return result

    def can_reach_target(self, target_board):
        '''
        functia care determina daca o stare duce catre o solutie.
        prima oara se verifica daca dimensiunile sunt mai mari decat
        dimensiunile solutiei, iar apoi se verifica daca solutia
        se afla printre caracterele starii
        '''

        if len(self.board) < len(target_board):
            return False

        if len(self.board[0]) < len(target_board[0]):
            return False

        last_target_line = 0
        for board_line in self.board:
            if last_target_line >= len(target_board):
                break
            if is_subseq(target_board[last_target_line], board_line):
                last_target_line = last_target_line + 1

        if last_target_line != len(target_board):
            return False

        return True

    def get_h(self, target_board, h_function, start_time, timeout):

        '''
        functia care calculeaza euristica. h_function este functia
        care se foloseste
        '''

        if not self.can_reach_target(target_board):    
            '''
            daca nu exista solutie, distanta estimata este infinit
            '''
            return float("inf")

        if h_function == 1:    
            '''
            prima functie este banala, returneaza 0 daca duce catre solutie, si infinit altfel 
            '''
            return 0

        elif h_function == 2:
            '''
            functie euristica admisibila care presupune ca distanta de la nodul curent
            la solutie este de o singura mutare, iar acea mutare este o mutare
            perfecta (nu exista caractere diferite, se elimina liniile/ coloanele deodata).
            Este admisibila deoarce ficare copil o sa adauge la cost o valoare cel putin egala
            cu cea calculata aici.
            '''
            if len(self.board) == len(target_board):
                return 1
                
            return min(1.0, (float(len(self.board[0])) / float(len(self.board) - len(target_board))))
        
        elif h_function == 3:
            '''
            functie euristica admisibila care returneaza costul vecinului cu cost cel mai mic
            minus costul curent. Adica presupune ca vecinul care are costul cel mai mic este solutia
            evident, acest cost este mai mic sau egal cu cel de la nodul curent pana la solutie.
            '''
            neighbours = self.get_neighbours(start_time, timeout)
            min_neighbour = None
            for neighbour in neighbours:
                if min_neighbour == None:
                    min_neighbour = neighbour
                else:
                    if min_neighbour.cost > neighbour.cost:
                        min_neighbour = neighbour
            
            return min_neighbour.cost - self.cost
        else:

            '''
            functie inadmisibila care presupune ca va lua un drum de cost maxim de la nodul
            curent la cel destinatie. (un drum cu un cost chiar mai mare decat cel maxim in multe cazuri)
            '''
            column_cost = float(len(self.board[0]) - len(target_board[0])) + float(len(self.board[0]) - len(target_board[0])) * (float(len(self.board) * 2.0)) 
            if len(self.board) == len(target_board):
                return column_cost

            row_cost = len(self.board[0]) * (len(self.board) - len(target_board))

            return column_cost + row_cost

    def get_neighbours(self, start_time, timeout):
        '''
        functia care returneaza nodurile la care se pot ajunge printr-o mutare
        de la configuratia curenta. Se foloseste backtracking pentru a testa
        toate configuratiile posibile.
        '''
        row_potential = []
        column_potential = []

        result = []
        
        self.backtracking(0, -1, len(self.board), [], row_potential, start_time, timeout)
        if len(self.board) > 0:
            self.backtracking(0, -1, len(self.board[0]), [], column_potential, start_time, timeout)

        for configuration in row_potential:
            new_board = [x for x in self.board]
            for cut in reversed(configuration):
                new_board.pop(cut)
            
            if len(self.board) > 0:
                col_count = len(self.board[0])
                result.append(Node(new_board, self.cost + float(col_count) / float(len(configuration)), self, configuration, None))

        for configuration in column_potential:
            new_board = []
            differences = 0
            for crt_line in self.board:
                new_line = crt_line
                for cut in reversed(configuration):
                    new_line = string_remove_at(cut, new_line)
                new_board.append(new_line)
                differences = differences + self.column_differences(cut, configuration)
            if len(configuration) < len(self.board[0]):
                result.append(Node(new_board, self.cost + 1 + (float(differences) / float(len(configuration))), self, None, configuration))            

        return result

@dataclass(order=True)
class PrioritizedItem:

    '''
    clasa auxiliara care ne ajuta sa folosim structura PriorityQueue din python pentru
    a optimiza codul si a nu folosi liste la UCS si A*. priority este prioritatea
    pe care o are obiectul (costul sau functia f dupa caz), iar item este nodul incapsulat
    '''

    priority: float
    item: object = field()

def is_solution(crt_node, target_node):

    '''
    functia care verifica daca am ajuns la o configuratie finala
    '''

    if crt_node == None:
        return False

    return crt_node.get_tuple_board() == target_node.get_tuple_board()

def UCS(initial_node, target_node, nsol, timeout):

    '''
    rezolva problema folosind UCS
    '''

    global processed_nodes
    processed_nodes = processed_nodes + 1

    q = PriorityQueue()
    q.put(PrioritizedItem(initial_node.cost, initial_node))
    visited = set()
    
    result = []
    start_time = time.time()

    while not q.empty():
        crt_node = q.get().item

        crt_node.nodes_in_mem = q.qsize()
        crt_node.elapsed_time = time.time() - start_time

        if is_solution(crt_node, target_node):
            if len(result) < nsol:
                result.append(crt_node)
            if len(result) >= nsol:
                return result

        if (time.time() - start_time) > timeout:
            return result

        if crt_node.get_tuple_board() in visited:
            continue

        visited.add(crt_node.get_tuple_board())

        neighbours = crt_node.get_neighbours(start_time, timeout)

        for neighbour in neighbours:
            processed_nodes = processed_nodes + 1
            
            q.put(PrioritizedItem(neighbour.cost, neighbour))
    
    return result

def a_star(initial_node, target_node, nsol, timeout, h_function):

    '''
    rezolva problema folosind A*
    '''

    global processed_nodes

    processed_nodes = processed_nodes + 1

    q = PriorityQueue()
    q.put(PrioritizedItem(0, initial_node))
    
    result = []
    start_time = time.time()

    while not q.empty():
        crt_node = q.get().item

        crt_node.nodes_in_mem = q.qsize()
        crt_node.elapsed_time = time.time() - start_time

        if is_solution(crt_node, target_node):
            if len(result) < nsol:
                result.append(crt_node)
            if len(result) >= nsol:
                return result

        if (time.time() - start_time) > timeout:
            return result

        neighbours = crt_node.get_neighbours(start_time, timeout)

        for neighbour in neighbours:

            processed_nodes = processed_nodes + 1

            if (time.time() - start_time) > timeout:
                return result

            q.put(PrioritizedItem(neighbour.cost + neighbour.get_h(target_node.board, h_function, start_time, timeout), neighbour))

    return result

def a_star_optimized(initial_node, target_node, nsol, timeout, h_function):

    '''
    rezolva problema folosind A* optimizat cu listele open si close
    si returneaza o singura solutie. In cazul asta algoritmul este mai lent decat
    A* implementat mai sus, deoarce aici nu am folosit o coada cu prioritati, ceea ce
    incetineste executarea programului.
    '''

    global processed_nodes
    processed_nodes = processed_nodes + 1
    
    open = []
    open.append(PrioritizedItem(0, initial_node))
    closed = set()

    result = []
    start_time = time.time()
    nsol = 1
    
    while len(open) != 0:
        crt_item = open[0]
        crt_node = open[0].item
        open.pop(0)

        crt_node.nodes_in_mem = len(open)
        crt_node.elapsed_time = time.time() - start_time

        if is_solution(crt_node, target_node):
            if len(result) < nsol:
                result.append(crt_node)
            if len(result) >= nsol:
                return result

        if (time.time() - start_time) > timeout:
            return result

        is_closed = False
        for closed_node in closed:
            if closed_node[1] == crt_node.get_tuple_board():
                is_closed = True

        if not is_closed:
            closed.add((crt_item.priority, crt_node.get_tuple_board()))

        neighbours = crt_node.get_neighbours(start_time, timeout)

        for neighbour in neighbours:

            processed_nodes = processed_nodes + 1

            if (time.time() - start_time) > timeout:
                return result

            open_position = -1
            ix = 0
            for open_node in open:
                if open_node.item.get_tuple_board() == neighbour.get_tuple_board():
                    open_position = ix
                ix = ix + 1

            closed_item = None
            for closed_node in closed:
                if closed_node[1] == neighbour.get_tuple_board():
                    closed_item = closed_node

            crt_f = neighbour.cost + neighbour.get_h(target_node.board, h_function, start_time, timeout)
            if open_position == -1 and closed_item == None:
                open.append(PrioritizedItem(crt_f, neighbour))
            else:
                if open_position != -1:
                    if crt_f < open[open_position].priority:
                        open[open_position].priority = crt_f
                else:
                    if crt_f < closed_item[0]:
                        closed.remove(closed_item)
                        closed.add((crt_f, neighbour.get_tuple_board()))
                        open.append(PrioritizedItem(crt_f, neighbour))
        
        open.sort()

    return result

def ida_star(initial_node, target_node, nsol, timeout, h_function):

    '''
    rezolva problema folosind IDA*
    '''
    
    start_time = time.time()

    threshold = initial_node.get_h(target_node.board, h_function, start_time, timeout)
    
    while True:
        solutions = []
        ret = search(initial_node, target_node, 0, threshold, solutions, nsol, h_function, timeout, start_time, 0)

        if is_solution(ret[1], target_node):
            if nsol >= len(solutions):
                return solutions[:nsol]

        if (time.time() - start_time) > timeout:
            return solutions

        if math.isinf(ret[0]):
            return []
        
        threshold = ret[0]

def search(crt_node, target_node, g, threshold, solutions, nsol, h_function, timeout, start_time, depth):

    '''
    functie de search folosita in IDA* care pargurge graful solutiilor pana
    thresholdul dat ca argument.
    '''

    global processed_nodes

    f = g + crt_node.get_h(target_node.board, h_function, start_time, timeout)

    processed_nodes = processed_nodes + 1

    crt_node.nodes_in_mem = depth
    crt_node.elapsed_time = time.time() - start_time

    if f > threshold:
        return (f, crt_node)
    if is_solution(crt_node, target_node):
        solutions.append(crt_node)
        return (f, crt_node)
    
    minm = (float("inf"), None)

    if (time.time() - start_time) > timeout:
        return minm

    neighbours = crt_node.get_neighbours(start_time, timeout)
    for neighbour in neighbours:
        ret = search(neighbour, target_node, neighbour.cost, threshold, solutions, nsol, h_function, timeout, start_time, depth + 1)
        if is_solution(ret[1], target_node) and len(solutions) >= nsol:
            return ret
        if ret[0] < minm[0]:
            minm = ret

        if (time.time() - start_time) > timeout:
            return minm

    return minm

def print_solution(solutions, fout):

    '''
    functie care primeste o lista de solutii si afiseaza pe ecran fiecare solutie
    asa cum este specificat in cerinte.
    '''
    
    global processed_nodes

    if len(solutions) == 0:
        fout.write("Nu s-a putut gasi o solutie.\n")

    ix = 1

    for solution_node in solutions:

        fout.write("Solutia " + str(ix) + ":\n")
        fout.write("Cost: " + str(solution_node.cost) + "\n")
        fout.write("Timp: " + str(solution_node.elapsed_time) + "\n\n")

        solution_list = []
        crt_node = solution_node
        while crt_node != None:
            solution_list.append(crt_node)
            crt_node = crt_node.prev_board

        node_ix = 1

        for solution in reversed(solution_list):

            if solution.cut_rows != None:
                print_str = "Am eliminat liniile "
                for i in range(0, len(solution.cut_rows)):
                    print_str = print_str + str(solution.cut_rows[i] + 1)
                    if i != len(solution.cut_rows) - 1:
                        print_str = print_str + ","
                print_str = print_str + "\n"
                fout.write(print_str + "\n")

            if (solution.cut_columns != None):
                print_str = "Am eliminat coloanele "
                for i in range(0, len(solution.cut_columns)):
                    print_str = print_str + str(solution.cut_columns[i] + 1)
                    if i != len(solution.cut_columns) - 1:
                        print_str = print_str + ","
                print_str = print_str + "\n"
                fout.write(print_str + "\n")

            fout.write("Node " + str(node_ix) + ":\n")
            fout.write("Numar noduri in memorie: " + str(solution.nodes_in_mem) + "\n\n")

            for line in solution.board:
                fout.write(line + "\n")

            node_ix = node_ix + 1
        ix = ix + 1
        fout.write("-------------------------------------\n")

    fout.write("Noduri calculate: " + str(processed_nodes) + "\n")

def main():

    global processed_nodes

    '''
    luam argumentele date din linia de comanda
    '''

    parser = ArgumentParser(usage=__file__ + ' '
                                             '-i/--folder input '
                                             '-o/--folder output'
                                             '-n/--numar solutii'
                                             '-t/--timp de timeout',
                            description='AI Homework 1')
    
    parser.add_argument('-i', '--folder input',
                        dest='input_folder',
                        default='',
                        help='Folderul in care se gasese fisierul input.txt.')

    parser.add_argument('-o', '--folder output',
                    dest='output_folder',
                    default='',
                    help='Folderul in care se vor scrie fisierele de output.')

    parser.add_argument('-n', '--numar solutii',
                    dest='nsol',
                    default=4,
                    help='Numarul de solutii afisate')

    parser.add_argument('-t', '--timeout',
                    dest='timeout',
                    default=1,
                    help='Numarul de secunde dupa care se da timeout')     

    args = vars(parser.parse_args())

    args["nsol"] = int(args["nsol"])
    args["timeout"] = int(args["timeout"])

    fin = open(os.path.join(args['input_folder'], "input.txt"), "r")
    lines = fin.readlines()    
    fin.close()

    initial_board = []
    target_board = []

    is_target = False
    prev_line = None
    valid_input = True

    for line in lines:
    
        '''
        parsam input-ul si ne asiguram ca este valid
        '''

        if line == '\n':
            prev_line = None
            is_target = True
            continue

        if not is_target:
            initial_board.append(line.replace("\n", ""))
        else:
            target_board.append(line.replace("\n", ""))

        if prev_line != None:
            if len(prev_line) != len(line.replace("\n", "")):
                print(line)
                valid_input = False
        
        if line != '\n':
            prev_line = line.replace("\n", "")

    if not valid_input:
        print("Invalid input.")
        return

    nsol = args['nsol']

    start_node = Node(initial_board, 0.0, None, None, None)
    end_node = Node(target_board, 0.0, None, None, None)

    if not start_node.can_reach_target(end_node.board):
        print("Nu exista solutie")
        return

    '''
    rulam algoritmii si scriem in fisiere rezultatele
    '''

    fout_ucs = open(os.path.join(args['output_folder'], "ucs.txt"), "w")
    processed_nodes = 0
    print_solution(UCS(Node(initial_board, 0.0, None, None, None), Node(target_board, 0.0, None, None, None), nsol, args["timeout"]), fout_ucs)
    fout_ucs.close()

    for euristhic in range(1, 5):

        fout_a_star = open(os.path.join(args['output_folder'], "a_star" + str(euristhic) + ".txt"), "w")
        processed_nodes = 0
        print_solution(a_star(Node(initial_board, 0.0, None, None, None), Node(target_board, 0.0, None, None, None), nsol, args["timeout"], euristhic), fout_a_star)
        fout_a_star.close()

        fout_a_star_optimized = open(os.path.join(args['output_folder'], "a_star_optimized" + str(euristhic) + ".txt"), "w")
        processed_nodes = 0
        print_solution(a_star_optimized(Node(initial_board, 0.0, None, None, None), Node(target_board, 0.0, None, None, None), nsol, args["timeout"], euristhic), fout_a_star_optimized)
        fout_a_star_optimized.close()

        fout_ida_star = open(os.path.join(args['output_folder'], "ida_star" + str(euristhic) + ".txt"), "w")
        processed_nodes = 0
        print_solution(ida_star(Node(initial_board, 0.0, None, None, None), Node(target_board, 0.0, None, None, None), nsol, args["timeout"], euristhic), fout_ida_star)
        fout_ida_star.close()

if __name__ == "__main__":
    main()