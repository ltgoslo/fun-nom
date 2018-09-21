
This repository includes the noun–noun compound dataset by 
[Fares (2016)](http://www.aclweb.org/anthology/P16-3011) which consists of
compounds annotated with two different taxonomies of relations; that is, for each
noun–noun compound there are two distinct relations, drawing on different
linguistic schools. The dataset was derived from existing linguistic resources,
such as [NomBank](https://nlp.cs.nyu.edu/meyers/NomBank.html) 
and [PCEDT 2.0](https://ufal.mff.cuni.cz/pcedt2.0/).
The noun–noun compounds themselves were extracted from the Wall Street Journal (WSJ)
portion in the Penn Treebank (PTB).

# The Dataset

The noun–noun compound dataset is available in two versions: type-based and
instance-based. The former assumes a type-based definition of the semantics 
of noun–noun compounds; that is, the relation holding between noun–noun
compounds is the same regardless of their context. Hence, the type-based dataset
includes one and only one instance of each compound.
The instance-based (or token-based) dataset, however, assumes that the semantics
of noun–noun compounds depend on the context they occur in, and therefore the 
dataset include multiple instance of the same compound _type_ which can be
annotated with different relations. 
Furthermore, the type- and instance-based include two files each: NNP0 and NNPH.
The NNP0 file includes the set of noun–noun compounds whose constituents
consist of common nouns only (no proper nouns are allowed in any position).
In the NNPH file, on the other hand, proper nouns are allowed but not in the
right-most constituent position.
For more information about this distinction, please refer to 
[Fares (2016)](http://www.aclweb.org/anthology/P16-3011).
Finally, the dataset includes both two-word and multi-word compounds, i.e.
compounds with two nominal constituents and compounds with more than two nominal
constituents.

### Type-based Dataset

The type-based dataset is organized in two tab-separated files (corresponding
to the treatment of proper nouns, NNP0 vs. NNPH). 
Each line in these files consists of three tab-separated columns, where the
first column is the compound itself, the second one is its NomBank relation and
the third is its PCEDT relation, e.g.: ``computer breakdown  ARG1   ACT-arg``

- [compound_types-NNP0.txt](compound_types-NNP0.txt): 10596 compounds
- [compound_types-NNPH.txt](compound_types-NNPH.txt): 14405 compounds



### Instance-based Dataset

The instance-based dataset is organized in two tab-separated files (corresponding
the NNP0 vs. NNPH distinction).
- [compound_instances-NNP0.txt](compound_instances-NNP0.txt): 21080 compounds
- [compound_instances-NNPH.txt](compound_instances-NNPH.txt): 26709 compounds


Each line consists of seven tab-separated columns as follows: 1) PTB WSJ
identifier, 2) sentence identifier, 3) index of the left-most constituent 
(within the sentence), 4) index of the right-most constituent, 5) noun–noun
compound instance, 6) NomBank relation and 7) PCEDT functor.

```
00/wsj_0003.mrg   0   11    12    cigarette filters   ARG2    RSTR
```

Assuming you have a local copy of the PTB, the context of the noun–noun
compounds (i.e. the sentences from which they were extracted) can be easily
retrieved using the first two columns in each line (the PTB WSJ identifier and
the sentence identifier). The dataset also provides the indices of the left- and
right-most constituents of each noun–noun compound within its sentence, to make
it possible to extract the surrounding window of _n_ words.

The following code snippet is a minimal example of how to extract the sentences 
(and reconstruct the compounds from their constituents' indices)

```python
from nltk.corpus import BracketParseCorpusReader
corpus_root = "/path/to/ldc/99T42/parsed/mrg/wsj"
file_pattern = ".*/wsj_.*\.mrg"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)

for line in open("compound_instances-NNP0.txt", 'r').readlines():
    columns = line.split('\t')
    sent_id = int(columns[1])
    from_id = int(columns[2])
    to_id = int(columns[3])

    sentence = ptb.sents(fileids=columns[0])[sent_id]
    constituents = []
    for i in range(from_id, to_id + 1):
        constituents.append(ptb.sents(fileids=columns[0])[sent_id][i])
    compound = ' '.join(constituents)
```


## How to Cite

If you use this dataset, please cite the following works:


```
@InProceedings{Fares:2016,
  author    = {Fares, Murhaf},
  title     =  {A {D}ataset for {J}oint {N}oun-{N}oun {C}ompound {B}racketing and {I}nterpretation},
  booktitle = {Proceedings of the {ACL} 2016 {S}tudent {R}esearch {W}orkshop},
  year      = {2016},
  publisher = {Association for Computational Linguistics},
  pages     =  {72--79},
  location  = {Berlin, Germany}
  doi       = {10.18653/v1/P16-3011},
  url       = {http://www.aclweb.org/anthology/P16-3011}
}
```


```
@inproceedings{Haj:Haj:Pan:12,
  booktitle = {Proceedings of the 8th {I}nternational {C}onference on {L}anguage {R}esources and {E}valuation ({LREC} 2012)},
  title     = {Announcing {P}rague {C}zech-{E}nglish {D}ependency {T}reebank 2.0},
  author    = {Jan Haji\v{c} and Eva Haji\v{c}ov{\'a} and Jarmila Panevov{\'a}
               and Petr Sgall and Ond\v{r}ej Bojar and Silvie Cinkov{\'a} 
               and Eva Fu\v{c}{\'\i}kov{\'a} and Marie Mikulov{\'a} 
               and Petr Pajas and Jan Popelka and Ji\v{r}{\'\i} Semeck{\'y} 
               and Jana \v{S}indlerov{\'a} and Jan \v{S}t\v{e}p{\'a}nek 
               and Josef Toman and Zde\v{n}ka Ure\v{s}ov{\'a} 
               and Zden\v{e}k \v{Z}abokrtsk{\'y}},
  year      = {2012},
  publisher = {{E}uropean {L}anguage {R}esources {A}ssociation},
  address   = {Istanbul, Turkey},
  pages     = {3153--3160},
  isbn      = {978-2-9517408-7-7},
}
```

```
@inproceedings{Mey:Ree:Mac:04,
  author    = {Meyers, Adam and Reeves, Ruth and Macleod, Catherine and 
            Szekely, Rachel and Zielinska, Veronika and Young, Brian and 
            Grishman, Ralph},
  year      = 2004,
  title     = {Annotating Noun Argument Structure for {N}om{B}ank},
  booktitle = {Proceedings of the 4th {I}nternational {C}onference on {L}anguage {R}esources and {E}valuation},
  address   = {Lisbon, Portugal},
  pages     = {803--806}
}   
```

## License
None of the morphosyntactic annotations (i.e. PoS tags and syntax trees) of the
PTB are included in the public release of this dataset.
The public version of the dataset only includes noun compound, out of
context, with their corresponding annotations from NomBank and PCEDT 2.0.
To obtain the full context from which the noun–noun compounds were extracted,
please refer to the description of the instance-based version of the dataset
above.
However, in order to retrieve the context you will need a local copy of the PTB 
(LDC99T42).

The NomBank relations were extracted from the NomBank version that is [available
in the public domain](https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0.zip). 
None of the LDC-licensed files in NomBank (i.e. _COMNOM.1.0_ and
_nombank.1.0.print_) were used.

The functor relations of the PCEDT were extracted from the PCEDT 2.0 dependency
annotation of the English data is [licensed under the terms of of CC-BY-NC-SA
3.0.](https://lindat.mff.cuni.cz/repository/xmlui/page/license-pcedt2)
 
The noun–noun compound dataset itself is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/).

