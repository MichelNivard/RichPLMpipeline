# Model and inference

The architecture is the established pre-norm protein Transformer plus current
single and pair heads. It embeds 23 sequence tokens, adds learned positions,
uses GELU Transformer encoder blocks, and normalizes the final representation.

| preset | `max_len` | `d_model` | layers | heads | FFN | pair rank | all-head parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100m` | 512 | 768 | 16 | 12 | 3072 | 128 | 115,669,163 |
| `300m` | 512 | 1024 | 24 | 16 | 4096 | 128 | 305,588,395 |

Counts include the PSSM/MLM projection, compatibility pair tensors, dense MI,
distance, contact, StructEncoder-latent, and 3Di heads. Tests allocate the
models on the meta device and compare exact counts, so configuration drift is a
test failure. The 100M schema remains compatible with mature 5M checkpoints;
older checkpoints without a 3Di head may load with only that head missing.

The pair heads project residues to rank 128, form a symmetric matrix product,
apply learned sequence-separation bias where used, and set diagonals to zero.
This avoids a learned dense pair-state trunk.

## What inference requires

The encoder accepts canonical amino-acid token IDs and an attention mask, with
at most 512 residues. For an embedding or MLM/mutation score it requires only
the sequence and checkpoint. It does not require UniClust, an MSA, AFDB,
Foldseek, coordinates, 3Di, or the StructEncoder teacher. Those are training
target sources.

ProteinGym masked-marginal scoring masks each sequence position, obtains the 20
amino-acid log probabilities, and scores a substitution as
`log p(mutant) - log p(wild type)`.

## Head semantics

- `pssm_logits`: shared 20-amino-acid readout for MLM and PSSM.
- `dense_mi_pred`: APC-MI regression in codec reconstruction units.
- `distance_pred`: log-distance regression (`log1p Å`).
- `contact_pred`: independent contact logits trained from distance labels.
- `struct_latent_pred`: normalized-teacher latent regression.
- `three_di_logits`: 23-class 3Di projection; training labels use 20 structural
  tokens and reserve padding/special IDs.
