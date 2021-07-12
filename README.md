ICFP Programming Contest 2021 - Team Sosuupoyo(sugoi) (one-person team) repository
====

Algorithm

Main routine: Simulated Annealing (SA)

To get optimization efficiency, invalid length of edges are allowed in intermediate state with penalty.
The penalty is increasing to get valid solution at last.

Another routine: Backtrack w/ SA

Some problem is known that the optimal cost is 0, I used backtrack method to assign figure's vertices to hole's vertices. Other vertices are optimized by SA.
