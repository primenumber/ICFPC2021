ICFP Programming Contest 2021 - Team Sosuupoyo(sugoi) (one-person team) repository
====

## Algorithm

### Main routine: Simulated Annealing (SA)

To get optimization efficiency, invalid length of edges are allowed in intermediate state with penalty.
The penalty is increasing to get valid solution at last.

SA using several neighborhoods:
- move one point slightly
- move one point to one of valid (hole limitation and edge limitation) positions
- If a vertex has exactly two adjacent vertices, move it to mirror of the two vertices.
- move one edge slightly
- rotate one edge slightly
- move a set of vertices in one direction
- move all vertices in one direction
- rotate all vertices slightly

To generate initial solution (may be invalid), move and rotate the figure randomly, if it fits the hole, use it.
When that failed many times (100k), shrink figure 0.9x and try to fit again and again.

For speedup annealing, skip some validation when the operation moves only one or two vertices.
And caching nearest point from each point of the hole for speedup.

### Another routine: Backtrack w/ SA

Some problem is known that the optimal cost is 0, I used backtrack method to assign figure's vertices to hole's vertices. Other vertices are optimized by SA.
