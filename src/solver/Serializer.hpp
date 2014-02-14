#ifndef SOLVER_SERIALIZER_HPP_
#define SOLVER_SERIALIZER_HPP_

#include <istream>                      // for istream, ostream
#include <memory>                       // for unique_ptr

#include "Observation.hpp"              // for Observation
#include "State.hpp"
#include "Solver.hpp"                   // for Solver

namespace solver {
class ActionNode;
class BeliefNode;
class BeliefTree;
class Histories;
class HistoryEntry;
class HistorySequence;
class ObservationMapping;
class StateInfo;
class StatePool;

class Serializer {
  public:
    /** Constructs a serializer for the given solver. */
    Serializer(Solver *solver) :
        solver_(solver) {
    }
    /** Default destructor. */
    virtual ~Serializer() = default;

    /* Copying and moving is disallowed. */
    Serializer(Serializer const &) = delete;
    Serializer(Serializer &&) = delete;
    Serializer &operator=(Serializer const &) = delete;
    Serializer &operator=(Serializer &&) = delete;

    /* --------------- Saving the entire solver. ----------------- */

    /** Saves the sate of the solver. */
    virtual void save(std::ostream &os) {
        save(*(solver_->allStates_), os);
        save(*(solver_->allHistories_), os);
        save(*(solver_->policy_), os);
    }
    /** Loads the state of the solver. */
    virtual void load(std::istream &is) {
        load(*(solver_->allStates_), is);
        load(*(solver_->allHistories_), is);
        load(*(solver_->policy_), is);
    }

    /* --------------- Saving states & observations ----------------- */

    /** Saves a State. */
    virtual void saveState(State const &state, std::ostream &os) = 0;
    /** Loads a State. */
    virtual std::unique_ptr<State> loadState(std::istream &is) = 0;

    /** Saves an observation. */
    virtual void saveObservation(Observation const &obs, std::ostream &os) = 0;
    /** Loads an Observation. */
    virtual std::unique_ptr<Observation> loadObservation(std::istream &is) = 0;

    /** Saves a mapping of observations to belief nodes. */
    virtual void saveMapping(ObservationMapping const &map, std::ostream &os) = 0;
    /** Loads a mapping of observations to belief nodes. */
    virtual std::unique_ptr<ObservationMapping> loadMapping(std::istream &is) = 0;

    /* --------------- Saving the state pool ----------------- */

    /** Saves a StateInfo. */
    virtual void save(StateInfo const &wrapper, std::ostream &os) = 0;
    /** Loads a StateInfo. */
    virtual void load(StateInfo &wrapper, std::istream &is) = 0;
    /** Saves a StatePool. */
    virtual void save(StatePool const &pool, std::ostream &os) = 0;
    /** Loads a StatePool. */
    virtual void load(StatePool &pool, std::istream &is) = 0;


    /* --------------- Saving the history sequences ----------------- */

    /** Saves a HistoryEntry. */
    virtual void save(HistoryEntry const &entry, std::ostream &os) = 0;
    /** Loads a HistoryEntry. */
    virtual void load(HistoryEntry &entry, std::istream &is) = 0;
    /** Saves a HistorySequence. */
    virtual void save(HistorySequence const &seq, std::ostream &os) = 0;
    /** Loads a HistorySequence. */
    virtual void load(HistorySequence &seq, std::istream &is) = 0;
    /** Saves a Histories. */
    virtual void save(Histories const &histories, std::ostream &os) = 0;
    /** Loads a Histories. */
    virtual void load(Histories &histories, std::istream &is) = 0;

    /* --------------- Saving the policy tree ----------------- */

    /** Saves an ActionNode. */
    virtual void save(ActionNode const &node, std::ostream &os) = 0;
    /** Loads an ActionNode. */
    virtual void load(ActionNode &node, std::istream &is) = 0;
    /** Saves a BeliefNode. */
    virtual void save(BeliefNode const &node, std::ostream &os) = 0;
    /** Loads a BeliefNode. */
    virtual void load(BeliefNode &node, std::istream &is) = 0;
    /** Saves a BeliefTree. */
    virtual void save(BeliefTree const &tree, std::ostream &os) = 0;
    /** Loads a BeliefTree. */
    virtual void load(BeliefTree &tree, std::istream &is) = 0;
  protected:
    Solver *solver_;
};
} /* namespace solver */

#endif /* SOLVER_SERIALIZER_HPP_ */