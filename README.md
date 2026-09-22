# NRP-Exp: Multi-Objective Experimentation and Decision-Space Analysis

NRP-Exp is an open-source Python platform for multi-objective Next Release
Problem experiments.

The platform integrates:

- JSON and synthetic instance provision.
- Evaluator-aware aggregation of multivalued attributes.
- Domain-specific preprocessing.
- Multi-objective solver execution.
- Pareto-front representation and comparison.
- Semantic enrichment and decision framing.
- Analytical lenses and Sets of Interest.
- Candidate Solution Set analysis.
- Interactive and programmatic workflows.

NRP-Exp is designed around explicit domain-plugin, solver, and analytical-lens
contracts. The current release provides one complete domain implementation for
the Next Release Problem. Solver and lens extensibility are demonstrated by
multiple implementations, while cross-domain reuse remains an architectural
extension point that has not yet been empirically evaluated.


Our platform bridges the gap between raw algorithmic optimization and human-centric decision-making by integrating data preprocessing, multi-objective solver execution, and interactive Pareto-front analysis into a single, cohesive workflow. It supports both interactive visual sessions via Streamlit and programmatic headless execution for batch experiments.

## Key Features

### NRP instance provision

NRP-Exp supports three entry routes:

- Structured NRP instances loaded from JSON.
- Synthetically generated NRP instances.
- Previously computed Pareto fronts imported from CSV.

JSON instances can include:

- Scalar and multivalued attributes.
- Stakeholder and developer evaluator groups.
- Evaluator weights.
- Precedence relationships.
- Requirement couplings.
- Mutual exclusions.
- Additive and multiplicative value dependencies.

### Ordered preprocessing pipeline

JSON models follow an explicit processing order:

1. Plugin-based extraction and model construction.
2. Resolution of multivalued attributes using evaluator weights and declared
   aggregation rules.
3. Coupling resolution through Disjoint-Set Union.
4. Value-dependency adjustment.
5. Exclusion branching and precedence-aware removal.

Coupled items are merged into unified decision items. Their scalar attributes
are combined using attribute-specific coupling rules, and references in
precedences, exclusions, and value dependencies are remapped to the new item
identifiers.

Value dependencies whose participants collapse into one unified item are
absorbed statically. Dependencies that continue to span several items remain
available for candidate-solution evaluation.

Mutual exclusions generate alternative subproblem branches. Removing one
endpoint of an exclusion also removes items that depend transitively on that
endpoint through precedence relationships.

### Multi-objective optimization

NRP-Exp currently provides six solvers through a common `BaseSolver` contract:

- Brute Force with feasibility pruning and an evaluation limit.
- Greedy Multi-Weight Sweep.
- NSGA-II.
- MILP Multi-Weight Sweep using `scipy.optimize.milp`.
- Random Search.
- Simulated Annealing.

Every solver receives an `OptimizationProblem` and returns a common
`ParetoFront` representation.

Brute-force enumeration is exact only when its evaluation limit permits the
complete feasible search space to be explored. MILP obtains an exact solution
for each weighted linear scalarization, but the weight sweep may omit
non-supported Pareto-optimal solutions.

### Pareto-front management

The common result model includes:

- `Solution`
- `ParetoFront`
- `SolutionSet`
- `SolutionSetCollection`

These abstractions support:

- Non-dominated solution representation.
- Objective-direction metadata.
- DataFrame conversion.
- JSON and CSV serialization.
- Named solution subsets.
- Provenance metadata.
- Storage and comparison of several fronts during an interactive session.

Front-level quality measures include:

- Number of solutions.
- Hypervolume.
- Spacing.
- Objective-space spread.

Meaningful comparison requires compatible objective names, directions, scaling
conventions, and hypervolume reference points.

### Post-optimization decision analysis

NRP-Exp supports a progressive transition from a Pareto front to a
deliberation-oriented Candidate Solution Set.

The analytical workflow includes:

- Semantic enrichment.
- Decision framing.
- Analytical lenses.
- Sets of Interest.
- Consensus analysis.
- Candidate Solution Set review.

Available analytical lenses include:

- **Manual**: identifier-based expert selection.
- **Diversity**: K-means, K-medoids, agglomerative clustering, and HDBSCAN,
  subject to dependency availability.
- **Efficiency**: benefit-cost ratios, normalized scores, and ideal-distance
  analysis.
- **Preference**: weighted sum, TOPSIS, VIKOR, and reference-point methods.
- **Indicator**: top-N and non-dominated filtering over semantic indicators.
- **Consensus**: threshold-based voting across previously saved Sets of
  Interest.

A Set of Interest records solutions selected through one analytical rationale.
A Candidate Solution Set contains alternatives retained for detailed
requirement-level and stakeholder-level comparison.

### Semantic enrichment

The NRP plugin can derive domain indicators from objective values and
requirement selections, including measures such as:

- Scope.
- Productivity.
- Effectiveness.
- Squandering.
- Dirtiness.
- Annoyance.

Stakeholder-coverage analysis uses original evaluator information to compare
requested and delivered requirements.

### Interactive visualization

The Streamlit interface provides:

- Interactive 2D and 3D scatter plots.
- Bubble plots.
- Distribution plots.
- Parallel-coordinate plots.
- Decision maps.
- Solution profile radar charts.
- Requirement-composition heatmaps.
- Stakeholder-coverage views.
- Original and working Pareto-front panels.
- Saved-front comparison.
- Candidate Solution Set analysis.

## Architecture

NRP-Exp separates interactive presentation from reusable computational
components.

```text
Streamlit interface
        |
        v
Application use cases
        |
        +--> Domain model and NRP plugin
        +--> Preprocessing engine
        +--> Optimization solvers
        +--> Analysis sessions and analytical lenses
```

The intended dependency direction is:

```text
interface
    -> application
        -> domain / solving / analysis
```

The application, domain, solving, and analysis packages must not depend on
Streamlit.

The current application layer exposes headless 



## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME/NRP-Exp.git](https://github.com/YOUR-USERNAME/NRP-Exp.git)
cd NRP-Exp
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Whether it is adding a new solver, creating a new analytical lens, or fixing a bug, please feel free to open an issue or submit a pull request.

