# Proposal feedback

Feedback from course staff on the proposal (submitted 20 Oct 2026), recorded 1 Nov 2026.

## Feedback

> This sounds like a great project! A couple of quick notes while you work on this. First I would ask if a heterogeneous graph is truly necessary or not. I think it would be interesting to try a bipartite graph between Firms and Advisors with LightGCN, but of course this is not required. A heterogeneous graph makes sense here, but it's always a compelling argument when you show exactly what it buys you. I would try to evaluate other graph techniques (does Attention help here via GAT, or is simple message passing good enough). Outside of that, I really like your stretch goal; a temporal GNN is a perfect fit here, and I hope you're able to get to it.

## Points raised

1. **Justify the heterogeneous graph.** Show what the adviser–firm–branch graph adds over a simpler graph. The suggestion is a bipartite adviser–firm graph with LightGCN (optional). Note that LightGCN can't embed unseen nodes (DECISIONS.md, Q19), and advisers and firms are new at each cut-off. A bipartite GraphSAGE would test the same question without that problem.
2. **Compare graph techniques.** Does attention (GAT) help, or is plain message passing (GraphSAGE) enough?
3. **Temporal GNN.** The stretch goal is seen as a good fit for the problem and is encouraged.
