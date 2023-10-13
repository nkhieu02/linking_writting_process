
Changes that make some diffences:
- no gradient clipping.
- increase the hidden size to 200. *
- padding and using pack_padded_sequence.
- maybe the decoding is hard => add one more relu and one more
linear layer.

What to try next:
- implement the transformer

Changes that does not make a huge differences:
- increase the embedding size
- implement the cosineannealing (at the momment)


Small tasks:


Trying:

- [] Implement the transformer