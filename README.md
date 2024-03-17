# LogicMP
It is the official repository of [LogicMP](https://arxiv.org/abs/2309.15458) where MP stands for message passing. LogicMP aims to combine neural networks (semantic representations) with first-order logic (symbolic knowledge). 


Note: The code for the relational graphs is included. Due to the privacy issues in Ant Group, the code for document images cannot be fully provided. Even though, I will show it is easy to reimplement.

# An Example: Transitivity rule in Document Images

## Task Definition
Task: Given the input image and input tokens, the task is to develop a function to predict whether two tokens coexist in a block. 
Rule: If tokens $i$ and $j$ are in the same block and tokens $j$ and $k$ are also together, then tokens $i$ and $k$ should be in the same block. Formally, $\forall i, j, k: \mathtt{C}(i, j) \wedge \mathtt{C}(j, k) \implies \mathtt{C}(i, k)$.

![Encoding transitivity rule in image understanding task](figures/logicmp-case.jpeg).

## Rules
