Investigation on beta.

Beta controls the possible number of nodes that can be re-inserted by the re-insertion operation. At first, the critical region contains sqrt(n) number of nodes. The re-insertion operation can re-insert (1-beta)*sqrt(n) nodes from the critical region. The remaining nodes after re-insertion will be the collective attacker. Therefore, at each iteration, the number of nodes in the collective attacker is at least beta*sqrt(n) nodes.

We evaluate values of Beta in {0.2, 0.4, 0.6, 0.8, 1.0} on three networks, humanDisease, facebook, condmat.

All experiments is executed 30 runs on each network with 8 threads. Running time is not exact single run time!


Conclusion from experimental result:
1. The running time increases as beta decreases.
2. There are no obvious improvement on solution quality over beta values except facebook. When beta sets 0.4, CA gets highest OFV on facebook.

For Design 1, beta is 1.
For Design 2, beta is 0.4.