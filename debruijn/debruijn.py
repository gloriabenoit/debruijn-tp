#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
import itertools
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as fastq:
        flag = -1
        sequence = ""
        for ligne in fastq:
            if ligne.startswith("@"):
                yield next(fastq).strip()


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_occ = {}
    for sequence in read_fastq(Path(fastq_file)):
        for kmer in cut_kmer(sequence, kmer_size):
            if kmer not in kmer_occ.keys():
                kmer_occ[kmer] = 1
            else:
                kmer_occ[kmer] +=1
    return kmer_occ


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = nx.DiGraph()
    for kmer, occurrence in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]

        graph.add_edge(prefix, suffix, weight=occurrence)

    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        # Conservation du premier
        if not delete_entry_node:
            path = path[1:]

        # Conservation du dernier
        if not delete_sink_node:
            path = path[:-1]

        graph.remove_nodes_from(path)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Test de la fréquence
    std = statistics.stdev(weight_avg_list)
    if std > 0:
        best_index = weight_avg_list.index(max(weight_avg_list))

    # Test de la longueur
    else:
        std = statistics.stdev(path_length)
        if std > 0:
            best_index = path_length.index(max(path_length))

        # Choix aléatoire
        else:
            best_index = randint(0, len(path_list) - 1)

    # Suppression des mauvais chemins
    path_list.pop(best_index)
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    return graph

def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    if nx.has_path(graph, ancestor_node, descendant_node):
        simple_paths = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))

    path_length = []
    weight_avg_list = []

    for path in simple_paths:
        path_length.append(len(path))
        weight_avg_list.append(path_average_weight(graph, path))

    graph = select_best_path(graph, simple_paths, path_length, weight_avg_list, False, False)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for node in graph.nodes:
        pred = list(graph.predecessors(node))
        if len(pred) > 1:
            for i, j in itertools.combinations(pred, 2):
                ancestor = nx.lowest_common_ancestor(graph, i, j)
                if ancestor != None:
                    bubble = True
                    break
            if bubble:
                keep_node = node
                break
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, keep_node))

    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        pred = list(graph.predecessors(node))
        if len(pred) > 1:
            path_list = []
            path_length = []
            weight_avg_list = []
            for entry in starting_nodes:
                if nx.has_path(graph, entry, node):
                    simple_paths = nx.all_simple_paths(graph, entry, node)

                    for path in simple_paths:
                        path_list.append(path)
                        path_length.append(len(path))
                        weight_avg_list.append(path_average_weight(graph, path))

            if len(path_list) > 1:
                graph = select_best_path(graph, path_list, path_length,
                                         weight_avg_list, True, False)
                return solve_entry_tips(graph, get_starting_nodes(graph))

    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        pred = list(graph.successors(node))
        if len(pred) > 1:
            path_list = []
            path_length = []
            weight_avg_list = []
            for end in ending_nodes:

                if nx.has_path(graph, node, end):
                    simple_paths = nx.all_simple_paths(graph, node, end)

                    for path in simple_paths:
                        path_list.append(path)
                        path_length.append(len(path))
                        weight_avg_list.append(path_average_weight(graph, path))

            if len(path_list) > 1:
                graph = select_best_path(graph, path_list, path_length,
                                         weight_avg_list, False, True)
                return solve_entry_tips(graph, get_sink_nodes(graph))

    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes:
        pred = list(graph.predecessors(node))
        if len(pred) == 0:
            starting_nodes.append(node)
    return starting_nodes

def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node in graph.nodes:
        pred = list(graph.successors(node))
        if len(pred) == 0:
            sink_nodes.append(node)
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start_node in starting_nodes:
        for end_node in ending_nodes:
            if nx.has_path(graph, start_node, end_node):
                simple_paths = nx.all_simple_paths(graph, start_node, end_node)

                for path in simple_paths:
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]

                    contigs.append((contig, len(contig)))
    return contigs

def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as output:
        for i, contig in enumerate(contigs_list):
            output.write(f">contig_{i} len={contig[1]}\n")

            formatted = textwrap.fill(contig[0], width=80)
            output.write(f"{formatted}\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Lecture des arguments
    fastq_file = args.fastq_file
    kmer_size = args.kmer_size

    # Construction du dictionnaire
    kmer_dict = build_kmer_dict(fastq_file, kmer_size)

    # Construction du graphe
    graph = build_graph(kmer_dict)

    # Résolution des bulles
    graph = simplify_bubbles(graph)

    # Résolution des pointes
    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))

    # Récupération des noeuds d'entrée
    starting_nodes = get_starting_nodes(graph)

    # Récupération des noeuds de sortie
    sink_nodes = get_sink_nodes(graph)

    # Ecriture des contigs
    contigs = get_contigs(graph, starting_nodes, sink_nodes)

    # Sauvegarde des contigs dans fichier
    save_contigs(contigs, Path("contigs_obtained.txt"))

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


    # Avec le BLAST, on retrouve bien 100% de match,
    # il nous manque juste le début de la séquence, probablement
    # à cause de notre technique pour enlever les tips.

if __name__ == "__main__":  # pragma: no cover
    main()
