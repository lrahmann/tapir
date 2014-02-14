#ifndef SOLVER_SOLVER_HPP_
#define SOLVER_SOLVER_HPP_

#include <memory>                       // for unique_ptr
#include <set>                          // for set
#include <utility>                      // for pair
#include <vector>                       // for vector

#include "global.hpp"                     // for RandomGenerator

#include "Action.hpp"                   // for Action
#include "ChangeFlags.hpp"               // for ChangeFlags
#include "Model.hpp"                    // for Model, Model::StepResult
#include "Observation.hpp"              // for Observation
#include "State.hpp"

namespace solver {
class BeliefNode;
class BeliefTree;
class Histories;
class HistoryEntry;
class HistorySequence;
class Serializer;
class StateInfo;
class StatePool;

class Solver {
  public:
    friend class Serializer;
    friend class TextSerializer;

    Solver(RandomGenerator *randGen, std::unique_ptr<Model> model);
    ~Solver();
    Solver(Solver const &) = delete;
    Solver(Solver &&) = delete;
    Solver &operator=(Solver const &) = delete;
    Solver &operator=(Solver &&) = delete;

    enum RolloutMode {
        ROLLOUT_RANDHEURISTIC = 0,
        ROLLOUT_POL = 1
    };

    /** Sets the serializer to be used by this solver. */
    void setSerializer(Serializer *serializer);

    /** Generates a starting policy for the solver, by generating the given
     * number (maxTrials) of episodes, and terminating episodes when the
     * current discount reaches the lowest allowed value (minimumDiscount).
     */
    void genPol(long maxTrials, double minimumDiscount);
    /** Runs a single simulation up to a maximum of nSteps steps, returning
     * the total discounted reward.
     *
     * Also sets trajSt, trajActId, trajObs, and trajRew to be the sequences
     * of states, actions, observations, and immediate rewards;
     *
     * actualNSteps will be the actual number of steps, which will be
     * different to nSteps if a terminal state is reached early.
     * totChTime will be the total amount of time spent on changing the model.
     * totImpTime will be the total amount of time spent on generating new
     * episodes to improve the policy.
     */
    double runSim(long nSteps, std::vector<long> &changeTimes,
            std::vector<std::unique_ptr<State>> &trajSt,
            std::vector<Action> &trajActId,
            std::vector<std::unique_ptr<Observation>> &trajObs,
            std::vector<double> &trajRew, long *actualNSteps, double *totChTime,
            double *totImpTime);

  private:
    /* ------------------ Episode sampling methods ------------------- */
    /** Searches from the root node for initial policy generation. */
    void singleSearch(double discountFactor, double minimumDiscount);
    /** Searches from the given start node with the given start state. */
    void singleSearch(BeliefNode *startNode, StateInfo *startStateInfo,
            long startDepth, double discountFactor, double minimumDiscount);
    /** Continues a pre-existing history sequence from its endpoint. */
    void continueSearch(HistorySequence *sequence,
            double discountFactor, double minimumDiscount);
    /** Performs a backup on the given history sequence. */
    void backup(HistorySequence *sequence);

    /** Uses a rollout method to select an action and get results. */
    std::pair<Model::StepResult, double> getRolloutAction(BeliefNode *belNode,
            State const &state, double startDiscount, double discountFactor);
    /** Attempts to find another belief node near enough to this one to be
     * suitable for the policy-based rollout heuristic.
     */
    BeliefNode *getNNBelNode(BeliefNode *b);
    /** Helper method for policy-based rollout. */
    double rolloutPolHelper(BeliefNode *currNode, State const &state, double disc);
    /** Updates the overall weighting of the different heuristic heuristics
     * based on their performance.
     */
    void updateHeuristicProbabilities(double valImprovement);

    /* ------------------ Simulation methods ------------------- */
    /** Simulates a single step. */
    Model::StepResult simAStep(BeliefNode *currentBelief, State const &currentState);
    /** Handles particle depletion during the simulation. */
    BeliefNode *addChild(BeliefNode *currNode, Action const &action,
            Observation const &obs,
            long timeStep);
    /** Improves the solution, with the root at the given node. */
    void improveSol(BeliefNode *startNode, long maxTrials,
            double minimumDiscount);

    /* ------------------ Methods for handling model changes ------------------- */
    void applyChanges();
    void updateSequence(HistorySequence *sequence);
    /** Negates a backup on the given history sequence. */
    void undoBackup(HistorySequence *sequence);

    /* ------------------ Private data fields ------------------- */
    /** The serializer to be used with this solver. */
    Serializer *serializer_;
    /** The random number generator used. */
    RandomGenerator *randGen_;
    /** The POMDP model */
    std::unique_ptr<Model> model_;
    /** The tree that stores the policy */
    std::unique_ptr<BeliefTree> policy_;
    /** The full collection of simulated histories. */
    std::unique_ptr<Histories> allHistories_;
    /** The pool of states. */
    std::unique_ptr<StatePool> allStates_;

    /** The rollout mode that was used in the last rollout. */
    RolloutMode lastRolloutMode_;
    /** The coefficient that determines how much to explore heuristics
     * vs. exploiting them.
     */
    double heuristicExploreCoefficient_;
    /** The time used by each rollout heuristic. */
    double timeUsedPerHeuristic_[2];
    /** The weight associated with each rollout heuristic. */
    double heuristicWeight_[2];
    /** The calculated probability for the usage of each rollout heuristic. */
    double heuristicProbability_[2];
    /** The number of times each rollout heuristic has been used. */
    long heuristicUseCount_[2];
};
} /* namespace solver */

#endif /* SOLVER_SOLVER_HPP_ */